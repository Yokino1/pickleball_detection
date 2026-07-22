"""
Court projector — image→court homography with quality checks.

Fails gracefully: returns None when keypoints are insufficient.
Never throws, never blocks the pipeline.

Quality gates (all must pass for a fresh homography):
1. ≥4 keypoints, left+right and near+far coverage
2. Not collinear (condition number < 1e5)
3. Convex hull area > min_area_px
4. RANSAC inlier ratio ≥ min_inlier_ratio
5. Mean reprojection error (in image pixels) ≤ max_reproj_error_px
6. Determinant > 1e-6

Cached homography expires after max_cached_frames.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .layout import (
    CANONICAL_KEYPOINTS,
    COURT_WIDTH_FT,
    COURT_LENGTH_FT,
    CENTER_X,
)

# Left-side keypoint indices: 0,3,6,8,11
_LEFT_INDICES  = {0, 3, 6, 8, 11}
# Right-side keypoint indices: 2,5,7,10,13
_RIGHT_INDICES = {2, 5, 7, 10, 13}
# Far-side (y ≤ 15): 0,1,2,3,4,5
_FAR_INDICES   = {0, 1, 2, 3, 4, 5}
# Near-side (y ≥ 29): 8,9,10,11,12,13
_NEAR_INDICES  = {8, 9, 10, 11, 12, 13}


class CourtProjector:
    """Compute and cache homography from court keypoint observations.

    Parameters
    ----------
    allow_cached : bool
    max_cached_frames : int
        Max frames to reuse a cached H before status → ``"unavailable"``.
    ransac_thresh : float
        RANSAC reprojection threshold in image pixels.
    min_inlier_ratio : float
        Minimum RANSAC inlier ratio (0–1).
    max_reproj_error_px : float
        Maximum mean reprojection error in **image pixels**.
    min_area_px : float
        Minimum convex-hull area of image keypoints (rejects degenerate sets).
    """

    def __init__(
        self,
        allow_cached: bool = True,
        max_cached_frames: int = 30,
        ransac_thresh: float = 5.0,
        min_inlier_ratio: float = 0.50,
        max_reproj_error_px: float = 8.0,
        min_area_px: float = 5000.0,
    ):
        self.allow_cached = allow_cached
        self.max_cached_frames = max_cached_frames
        self.ransac_thresh = ransac_thresh
        self.min_inlier_ratio = min_inlier_ratio
        self.max_reproj_error = max_reproj_error_px
        self.min_area_px = min_area_px

        self._H: Optional[np.ndarray] = None
        self._H_cached: Optional[np.ndarray] = None
        self._status: str = "none"
        self._cached_frames: int = 0
        self._quality: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._H

    @property
    def is_available(self) -> bool:
        return self._H is not None

    @property
    def quality(self) -> dict:
        return dict(self._quality)

    def update(self, observation) -> str:
        """Update homography. Returns ``"homography"`` | ``"cached"`` | ``"unavailable"``."""
        src, dst, skip_reason, kp_indices = self._prepare_points(observation)

        if src is not None and len(src) >= 4:
            H, quality = self._compute_homography(src, dst, kp_indices)
            if H is not None:
                self._H = H
                self._H_cached = H
                self._cached_frames = 0
                self._status = "homography"
                self._quality = quality
                return self._status
            self._quality = quality
        else:
            self._quality = {
                "skip_reason": skip_reason,
                "num_points": len(src) if src is not None else 0,
                "status": "rejected",
            }

        # Fall back to cache
        if self.allow_cached and self._H_cached is not None:
            self._cached_frames += 1
            if self._cached_frames <= self.max_cached_frames:
                self._H = self._H_cached
                self._status = "cached"
                self._quality["cache_age"] = self._cached_frames
                return self._status
            self._H = None
            self._status = "unavailable"
            self._quality["cache_age"] = self._cached_frames
            return self._status

        self._H = None
        self._status = "unavailable"
        return self._status

    def project_missing_keypoints(self, observation) -> dict[int, tuple[float, float]]:
        """Project canonical keypoints missing from *observation* via H⁻¹.

        Returns a dict of {kp_idx: (x_px, y_px)} for points that are
        in CANONICAL_KEYPOINTS but not yet in the observation.
        Points may be outside the image bounds — callers should handle that.
        """
        if self._H is None:
            return {}
        try:
            H_inv = np.linalg.inv(self._H)
        except np.linalg.LinAlgError:
            return {}

        projected = {}
        for kp_idx, (cx, cy) in CANONICAL_KEYPOINTS.items():
            if kp_idx in observation.keypoints:
                continue
            pt = H_inv @ np.array([cx, cy, 1.0])
            if abs(pt[2]) < 1e-8:
                continue
            px, py = float(pt[0] / pt[2]), float(pt[1] / pt[2])
            projected[kp_idx] = (px, py)
        return projected

    def project(self, image_pt: tuple[float, float]) -> Optional[tuple[float, float]]:
        if self._H is None:
            return None
        p = np.array([[[float(image_pt[0]), float(image_pt[1])]]], dtype=np.float32)
        try:
            projected = cv2.perspectiveTransform(p, self._H)
            x, y = float(projected[0][0][0]), float(projected[0][0][1])
            if x < -20 or x > COURT_WIDTH_FT + 20 or y < -20 or y > COURT_LENGTH_FT + 20:
                return None
            return (x, y)
        except cv2.error:
            return None

    def reset(self) -> None:
        self._H = None
        self._H_cached = None
        self._status = "none"
        self._cached_frames = 0
        self._quality = {}

    # ------------------------------------------------------------------
    # Internal: point preparation + geometry checks
    # ------------------------------------------------------------------

    def _prepare_points(self, observation):
        """Build src/dst and run geometry distribution checks.

        Returns (src, dst, skip_reason, kp_indices).
        """
        if observation is None:
            return None, None, "no_observation", []

        src_pts, dst_pts, kp_idx_list = [], [], []
        for kp_idx, (px, py) in observation.keypoints.items():
            if kp_idx in CANONICAL_KEYPOINTS:
                src_pts.append([px, py])
                dst_pts.append(list(CANONICAL_KEYPOINTS[kp_idx]))
                kp_idx_list.append(kp_idx)

        if len(src_pts) < 4:
            return None, None, f"too_few:{len(src_pts)}", kp_idx_list

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        idx_set = set(kp_idx_list)

        # Distribution: must cover left+right AND near+far
        has_left  = bool(idx_set & _LEFT_INDICES)
        has_right = bool(idx_set & _RIGHT_INDICES)
        has_near  = bool(idx_set & _NEAR_INDICES)
        has_far   = bool(idx_set & _FAR_INDICES)

        if not (has_left and has_right):
            return None, None, "missing_left_right_coverage", kp_idx_list
        if not (has_near and has_far):
            return None, None, "missing_near_far_coverage", kp_idx_list

        # Collinearity: condition number of centred points
        centred = src - src.mean(axis=0)
        if len(centred) >= 3:
            _, s, _ = np.linalg.svd(centred)
            cond = s[0] / (s[1] + 1e-10)
            if cond > 1e5:
                return None, None, "collinear_points", kp_idx_list

        # Convex hull area
        hull = cv2.convexHull(src.astype(np.float32))
        area = cv2.contourArea(hull)
        if area < self.min_area_px:
            return None, None, f"area_too_small:{area:.0f}", kp_idx_list

        return src, dst, None, kp_idx_list

    # ------------------------------------------------------------------
    # Internal: homography computation + quality
    # ------------------------------------------------------------------

    def _compute_homography(self, src, dst, kp_indices):
        """Compute H with quality checks. Returns (H, quality_dict)."""
        quality = {
            "num_points": len(src),
            "inlier_ratio": 0.0,
            "mean_reproj_error_px": 999.0,
            "max_reproj_error_px": 999.0,
            "area_px": 0.0,
            "status": "rejected",
        }

        try:
            H, mask = cv2.findHomography(
                src, dst, cv2.RANSAC, self.ransac_thresh,
            )
        except cv2.error:
            return None, quality

        if H is None:
            return None, quality

        # Determinant
        if abs(np.linalg.det(H)) < 1e-6:
            return None, quality

        # Inlier ratio
        if mask is not None:
            inlier_count = int(np.sum(mask))
            quality["inlier_ratio"] = round(inlier_count / len(src), 3)
            if quality["inlier_ratio"] < self.min_inlier_ratio:
                return None, quality

        # Reprojection error — project dst (feet) back to image (px)
        # Compare with original src (px) → error in image pixels
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None, quality

        dst_proj_back = cv2.perspectiveTransform(
            dst.reshape(-1, 1, 2), H_inv
        ).reshape(-1, 2)
        errors_px = np.linalg.norm(dst_proj_back - src, axis=1)
        mean_px = float(np.mean(errors_px))
        max_px  = float(np.max(errors_px))
        quality["mean_reproj_error_px"] = round(mean_px, 2)
        quality["max_reproj_error_px"]  = round(max_px, 2)

        # Per-point diagnostics — keep what was already computed
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), dtype=bool)
        per_point = []
        for i, kp_idx in enumerate(kp_indices):
            per_point.append({
                "kp_idx": kp_idx,
                "image_xy": [round(float(src[i, 0]), 1), round(float(src[i, 1]), 1)],
                "court_ft": [round(float(dst[i, 0]), 1), round(float(dst[i, 1]), 1)],
                "reproj_error_px": round(float(errors_px[i]), 2),
                "is_inlier": bool(inlier_mask[i]),
            })
        quality["per_point"] = per_point

        # Hull area
        hull = cv2.convexHull(src.astype(np.float32))
        quality["area_px"] = round(float(cv2.contourArea(hull)), 0)

        if mean_px > self.max_reproj_error:
            quality["status"] = "rejected_reproj_error"
            return None, quality

        quality["status"] = "accepted"
        return H, quality
