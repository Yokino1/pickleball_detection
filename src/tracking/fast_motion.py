"""Conservative high-speed motion proposals for short YOLO gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import BallDetection


@dataclass
class _MotionBlob:
    center: tuple[float, float]
    bbox: list[float]
    area: float
    streak: int = 1


class FastMotionProposalGenerator:
    """Generate ball-sized motion candidates only inside an activated ROI.

    This is not a second general-purpose detector. It stays dormant unless a
    tracker prediction or cross-camera handoff provides a search ROI, and a
    candidate must move quickly across consecutive frames before it is emitted.
    """

    def __init__(
        self,
        analysis_width: int = 640,
        difference_threshold: int = 24,
        min_component_area: float = 2.0,
        max_component_area: float = 180.0,
        max_component_dimension: int = 48,
        max_aspect_ratio: float = 6.0,
        min_speed_px_per_second: float = 180.0,
        max_speed_px_per_second: float = 3200.0,
        reference_frame_width: float = 1280.0,
        frame_scale_override: Optional[float] = None,
        min_streak: int = 2,
        confidence: float = 0.21,
        max_proposals: int = 4,
        max_global_motion_fraction: float = 0.20,
        blur_kernel: int = 3,
        dilation_iterations: int = 1,
    ):
        self.analysis_width = max(160, int(analysis_width))
        self.difference_threshold = min(255, max(1, int(difference_threshold)))
        self.min_component_area = max(1.0, float(min_component_area))
        self.max_component_area = max(
            self.min_component_area, float(max_component_area)
        )
        self.max_component_dimension = max(2, int(max_component_dimension))
        self.max_aspect_ratio = max(1.0, float(max_aspect_ratio))
        self.min_speed_px_per_second = max(0.0, float(min_speed_px_per_second))
        self.max_speed_px_per_second = max(
            self.min_speed_px_per_second, float(max_speed_px_per_second)
        )
        self.reference_frame_width = max(1.0, float(reference_frame_width))
        self.frame_scale_override = (
            None
            if frame_scale_override is None
            else max(0.25, float(frame_scale_override))
        )
        self.min_streak = max(1, int(min_streak))
        self.confidence = min(1.0, max(0.0, float(confidence)))
        self.max_proposals = max(1, int(max_proposals))
        self.max_global_motion_fraction = min(
            1.0, max(0.0, float(max_global_motion_fraction))
        )
        kernel = max(1, int(blur_kernel))
        self.blur_kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.dilation_iterations = max(0, int(dilation_iterations))
        self._previous_gray: Optional[np.ndarray] = None
        self._previous_timestamp_s: Optional[float] = None
        self._previous_blobs: list[_MotionBlob] = []
        self._last_diagnostics = self._empty_diagnostics()

    @property
    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def frame_scale(self, frame_width: int) -> float:
        if self.frame_scale_override is not None:
            return self.frame_scale_override
        return max(0.25, float(frame_width) / self.reference_frame_width)

    def propose(
        self,
        frame: np.ndarray,
        *,
        timestamp_s: Optional[float],
        search_roi: Optional[list[float]],
        camera_motion: tuple[float, float] = (0.0, 0.0),
    ) -> list[BallDetection]:
        gray, analysis_scale = self._prepare(frame)
        previous = self._previous_gray
        previous_timestamp_s = self._previous_timestamp_s
        self._previous_gray = gray
        self._previous_timestamp_s = timestamp_s

        enabled = search_roi is not None
        if previous is None or previous.shape != gray.shape:
            self._previous_blobs = []
            self._last_diagnostics = {
                **self._empty_diagnostics(),
                "enabled": enabled,
                "warmup": True,
                "search_roi": search_roi,
            }
            return []
        if not enabled:
            self._previous_blobs = []
            self._last_diagnostics = self._empty_diagnostics()
            return []

        dx = float(camera_motion[0]) * analysis_scale
        dy = float(camera_motion[1]) * analysis_scale
        aligned_previous = cv2.warpAffine(
            previous,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (gray.shape[1], gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        difference = cv2.absdiff(aligned_previous, gray)
        _, motion_mask = cv2.threshold(
            difference,
            self.difference_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        if self.dilation_iterations > 0:
            motion_mask = cv2.dilate(
                motion_mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=self.dilation_iterations,
            )

        roi_mask = np.zeros_like(motion_mask)
        x1, y1, x2, y2 = self._scaled_roi(
            search_roi,
            analysis_scale,
            motion_mask.shape[1],
            motion_mask.shape[0],
        )
        roi_mask[y1:y2, x1:x2] = motion_mask[y1:y2, x1:x2]
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        motion_fraction = float(np.count_nonzero(roi_mask)) / float(roi_area)
        if (
            self.max_global_motion_fraction > 0.0
            and motion_fraction > self.max_global_motion_fraction
        ):
            self._previous_blobs = []
            self._last_diagnostics = {
                **self._empty_diagnostics(),
                "enabled": True,
                "fail_closed": True,
                "motion_fraction": round(motion_fraction, 4),
                "search_roi": [float(value) for value in search_roi],
            }
            return []

        blobs = self._extract_blobs(roi_mask, analysis_scale)
        dt_s = self._resolve_dt(timestamp_s, previous_timestamp_s)
        frame_scale = self.frame_scale(frame.shape[1])
        min_distance = self.min_speed_px_per_second * frame_scale * dt_s
        max_distance = self.max_speed_px_per_second * frame_scale * dt_s
        matched_blobs: list[_MotionBlob] = []
        for blob in blobs:
            previous_blob = self._nearest_previous(
                blob,
                min_distance=min_distance,
                max_distance=max_distance,
            )
            if previous_blob is not None:
                blob.streak = previous_blob.streak + 1
            matched_blobs.append(blob)
        self._previous_blobs = matched_blobs

        eligible = [
            blob for blob in matched_blobs if blob.streak >= self.min_streak
        ]
        eligible.sort(key=lambda blob: (-blob.streak, blob.area))
        proposals = [
            BallDetection(
                bbox=list(blob.bbox),
                center=[float(blob.center[0]), float(blob.center[1])],
                confidence=self.confidence,
                source="fast_motion",
            )
            for blob in eligible[: self.max_proposals]
        ]
        self._last_diagnostics = {
            "enabled": True,
            "warmup": False,
            "fail_closed": False,
            "motion_fraction": round(motion_fraction, 4),
            "search_roi": [float(value) for value in search_roi],
            "component_count": len(blobs),
            "streak_count": len(eligible),
            "proposal_count": len(proposals),
            "dt_ms": round(dt_s * 1000.0, 3),
        }
        return proposals

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_timestamp_s = None
        self._previous_blobs = []
        self._last_diagnostics = self._empty_diagnostics()

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.analysis_width / float(gray.shape[1]))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (
                    self.analysis_width,
                    max(1, int(round(gray.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        if self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return gray, scale

    @staticmethod
    def _scaled_roi(
        roi: list[float],
        scale: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(np.floor(float(roi[0]) * scale))))
        y1 = max(0, min(height - 1, int(np.floor(float(roi[1]) * scale))))
        x2 = max(0, min(width, int(np.ceil(float(roi[2]) * scale))))
        y2 = max(0, min(height, int(np.ceil(float(roi[3]) * scale))))
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)

    def _extract_blobs(
        self,
        motion_mask: np.ndarray,
        analysis_scale: float,
    ) -> list[_MotionBlob]:
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        blobs: list[_MotionBlob] = []
        inverse_scale = 1.0 / max(analysis_scale, 1e-6)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_component_area or area > self.max_component_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if max(width, height) > self.max_component_dimension:
                continue
            aspect = max(width, height) / float(max(1, min(width, height)))
            if aspect > self.max_aspect_ratio:
                continue
            bbox = [
                x * inverse_scale,
                y * inverse_scale,
                (x + width) * inverse_scale,
                (y + height) * inverse_scale,
            ]
            blobs.append(
                _MotionBlob(
                    center=(
                        (x + width / 2.0) * inverse_scale,
                        (y + height / 2.0) * inverse_scale,
                    ),
                    bbox=bbox,
                    area=area,
                )
            )
        return blobs

    def _nearest_previous(
        self,
        blob: _MotionBlob,
        *,
        min_distance: float,
        max_distance: float,
    ) -> Optional[_MotionBlob]:
        candidates = []
        for previous in self._previous_blobs:
            distance = float(
                np.hypot(
                    blob.center[0] - previous.center[0],
                    blob.center[1] - previous.center[1],
                )
            )
            if min_distance <= distance <= max_distance:
                candidates.append((distance, previous))
        return min(candidates, key=lambda item: item[0], default=(0.0, None))[1]

    @staticmethod
    def _resolve_dt(
        timestamp_s: Optional[float],
        previous_timestamp_s: Optional[float],
    ) -> float:
        if timestamp_s is None or previous_timestamp_s is None:
            return 1.0 / 30.0
        dt_s = float(timestamp_s) - float(previous_timestamp_s)
        return dt_s if np.isfinite(dt_s) and dt_s > 0.0 else 1.0 / 30.0

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {
            "enabled": False,
            "warmup": False,
            "fail_closed": False,
            "motion_fraction": 0.0,
            "search_roi": None,
            "component_count": 0,
            "streak_count": 0,
            "proposal_count": 0,
            "dt_ms": 0.0,
        }
