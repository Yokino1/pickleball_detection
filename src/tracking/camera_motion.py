"""Estimate frame-to-frame camera translation from background features."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMotion:
    dx: float = 0.0
    dy: float = 0.0
    tracked_points: int = 0
    inliers: int = 0
    confidence: float = 0.0
    applied: bool = False


class CameraMotionEstimator:
    """Estimate robust global translation using sparse optical flow and RANSAC."""

    def __init__(
        self,
        analysis_width: int = 640,
        max_corners: int = 300,
        min_inliers: int = 20,
        max_shift_px: float = 80.0,
        ransac_threshold_px: float = 2.5,
    ):
        self.analysis_width = max(160, int(analysis_width))
        self.max_corners = max(40, int(max_corners))
        self.min_inliers = max(6, int(min_inliers))
        self.max_shift_px = max(0.0, float(max_shift_px))
        self.ransac_threshold_px = max(0.5, float(ransac_threshold_px))
        self._previous_gray: np.ndarray | None = None
        self._last_motion = CameraMotion()

    @property
    def diagnostics(self) -> dict:
        motion = self._last_motion
        return {
            "dx": motion.dx,
            "dy": motion.dy,
            "tracked_points": motion.tracked_points,
            "inliers": motion.inliers,
            "confidence": motion.confidence,
            "applied": motion.applied,
        }

    def update(self, frame: np.ndarray) -> CameraMotion:
        gray, scale = self._prepare(frame)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._last_motion = CameraMotion()
            return self._last_motion

        previous = self._previous_gray
        self._previous_gray = gray
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )
        if points is None or len(points) < self.min_inliers:
            self._last_motion = CameraMotion()
            return self._last_motion

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or status is None:
            self._last_motion = CameraMotion()
            return self._last_motion

        valid = status.reshape(-1).astype(bool)
        old = points.reshape(-1, 2)[valid]
        new = current_points.reshape(-1, 2)[valid]
        tracked = len(old)
        if tracked < self.min_inliers:
            self._last_motion = CameraMotion(tracked_points=tracked)
            return self._last_motion

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px,
            maxIters=1000,
            confidence=0.99,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        confidence = inliers / tracked if tracked else 0.0
        if matrix is None or inliers < self.min_inliers:
            self._last_motion = CameraMotion(
                tracked_points=tracked,
                inliers=inliers,
                confidence=confidence,
            )
            return self._last_motion

        dx = float(matrix[0, 2] / scale)
        dy = float(matrix[1, 2] / scale)
        magnitude = float(np.hypot(dx, dy))
        applied = self.max_shift_px == 0.0 or magnitude <= self.max_shift_px
        self._last_motion = CameraMotion(
            dx=dx if applied else 0.0,
            dy=dy if applied else 0.0,
            tracked_points=tracked,
            inliers=inliers,
            confidence=confidence,
            applied=applied,
        )
        return self._last_motion

    def reset(self) -> None:
        self._previous_gray = None
        self._last_motion = CameraMotion()

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1]
        if width <= self.analysis_width:
            return gray, 1.0
        scale = self.analysis_width / width
        resized = cv2.resize(
            gray,
            (self.analysis_width, max(1, int(round(gray.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale
