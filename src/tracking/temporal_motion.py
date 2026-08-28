"""Lightweight consecutive-frame motion evidence for ball detections."""

from __future__ import annotations

import cv2
import numpy as np

from .types import BallDetection


class TemporalMotionFilter:
    """Reject detector candidates without local frame-to-frame motion.

    The previous grayscale frame is translated into the current frame using
    the camera-motion estimate before differencing. The filter fails open when
    it has no history or when too much of the frame changes at once.
    """

    def __init__(
        self,
        analysis_width: int = 640,
        difference_threshold: int = 18,
        min_motion_fraction: float = 0.015,
        region_scale: float = 1.5,
        min_region_radius_px: int = 8,
        max_global_motion_fraction: float = 0.35,
        blur_kernel: int = 5,
        dilation_iterations: int = 2,
    ):
        self.analysis_width = max(160, int(analysis_width))
        self.difference_threshold = min(255, max(1, int(difference_threshold)))
        self.min_motion_fraction = min(1.0, max(0.0, float(min_motion_fraction)))
        self.region_scale = max(0.5, float(region_scale))
        self.min_region_radius_px = max(2, int(min_region_radius_px))
        self.max_global_motion_fraction = min(
            1.0, max(0.0, float(max_global_motion_fraction))
        )
        kernel = max(1, int(blur_kernel))
        self.blur_kernel = kernel if kernel % 2 == 1 else kernel + 1
        self.dilation_iterations = max(0, int(dilation_iterations))
        self._previous_gray: np.ndarray | None = None
        self._last_diagnostics = self._empty_diagnostics()

    @property
    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def filter(
        self,
        frame: np.ndarray,
        detections: list[BallDetection],
        camera_motion: tuple[float, float] = (0.0, 0.0),
    ) -> list[BallDetection]:
        gray, scale = self._prepare(frame)
        previous = self._previous_gray
        self._previous_gray = gray

        if previous is None or previous.shape != gray.shape:
            self._last_diagnostics = {
                **self._empty_diagnostics(),
                "warmup": True,
                "input_detections": len(detections),
                "accepted_detections": len(detections),
            }
            return list(detections)

        dx = float(camera_motion[0]) * scale
        dy = float(camera_motion[1]) * scale
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

        global_motion_fraction = float(np.count_nonzero(motion_mask)) / float(
            motion_mask.size
        )
        fail_open = (
            self.max_global_motion_fraction > 0.0
            and global_motion_fraction > self.max_global_motion_fraction
        )

        accepted: list[BallDetection] = []
        scores: list[dict] = []
        for detection in detections:
            motion_fraction = self._motion_fraction(motion_mask, detection, scale)
            keep = fail_open or motion_fraction >= self.min_motion_fraction
            if keep:
                accepted.append(detection)
            scores.append(
                {
                    "center": [
                        round(float(detection.center[0]), 3),
                        round(float(detection.center[1]), 3),
                    ],
                    "confidence": round(float(detection.confidence), 4),
                    "motion_fraction": round(motion_fraction, 4),
                    "accepted": keep,
                }
            )

        self._last_diagnostics = {
            "warmup": False,
            "fail_open": fail_open,
            "global_motion_fraction": round(global_motion_fraction, 4),
            "input_detections": len(detections),
            "accepted_detections": len(accepted),
            "rejected_detections": len(detections) - len(accepted),
            "detection_scores": scores,
        }
        return accepted

    def reset(self) -> None:
        self._previous_gray = None
        self._last_diagnostics = self._empty_diagnostics()

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = gray.shape[1]
        scale = min(1.0, self.analysis_width / float(width))
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
            gray = cv2.GaussianBlur(
                gray,
                (self.blur_kernel, self.blur_kernel),
                0,
            )
        return gray, scale

    def _motion_fraction(
        self,
        motion_mask: np.ndarray,
        detection: BallDetection,
        scale: float,
    ) -> float:
        cx = float(detection.center[0]) * scale
        cy = float(detection.center[1]) * scale
        width = max(1.0, float(detection.bbox[2] - detection.bbox[0])) * scale
        height = max(1.0, float(detection.bbox[3] - detection.bbox[1])) * scale
        radius = max(
            float(self.min_region_radius_px),
            0.5 * max(width, height) * self.region_scale,
        )
        x1 = max(0, int(np.floor(cx - radius)))
        y1 = max(0, int(np.floor(cy - radius)))
        x2 = min(motion_mask.shape[1], int(np.ceil(cx + radius + 1)))
        y2 = min(motion_mask.shape[0], int(np.ceil(cy + radius + 1)))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = motion_mask[y1:y2, x1:x2]
        return float(np.count_nonzero(region)) / float(region.size)

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {
            "warmup": False,
            "fail_open": False,
            "global_motion_fraction": 0.0,
            "input_detections": 0,
            "accepted_detections": 0,
            "rejected_detections": 0,
            "detection_scores": [],
        }
