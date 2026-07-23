"""Detection-only ball tracking pipeline, independent of court projection."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .ball_detector import BallDetector
from .camera_motion import CameraMotionEstimator
from .multi_ball_tracker import MultiBallTracker
from .types import FrameResult


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_ball_detections(
    detections,
    iou_threshold: float = 0.20,
    center_scale: float = 0.75,
    minimum_center_px: float = 6.0,
):
    """Keep the strongest box from each ball-sized cluster."""
    kept = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        width = max(1.0, detection.bbox[2] - detection.bbox[0])
        height = max(1.0, detection.bbox[3] - detection.bbox[1])
        duplicate = False
        for accepted in kept:
            accepted_width = max(1.0, accepted.bbox[2] - accepted.bbox[0])
            accepted_height = max(1.0, accepted.bbox[3] - accepted.bbox[1])
            center_distance = float(
                np.hypot(
                    detection.center[0] - accepted.center[0],
                    detection.center[1] - accepted.center[1],
                )
            )
            center_limit = max(
                minimum_center_px,
                center_scale * max(width, height, accepted_width, accepted_height),
            )
            if _bbox_iou(detection.bbox, accepted.bbox) >= iou_threshold or center_distance <= center_limit:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


class BallTrackingPipeline:
    """Run full-frame detection and multi-ball tracking for each video frame."""

    def __init__(
        self,
        detector: BallDetector,
        tracker: MultiBallTracker,
        detector_interval: int = 1,
        duplicate_iou_threshold: float = 0.20,
        duplicate_center_scale: float = 0.75,
        duplicate_center_px: float = 6.0,
        camera_motion_estimator: CameraMotionEstimator | None = None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.detector_interval = max(1, int(detector_interval))
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.duplicate_center_scale = float(duplicate_center_scale)
        self.duplicate_center_px = float(duplicate_center_px)
        self.camera_motion_estimator = camera_motion_estimator
        self._processed_frames = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: Optional[float] = None,
    ) -> FrameResult:
        height, width = frame.shape[:2]
        camera_started = time.perf_counter()
        camera_motion = (
            self.camera_motion_estimator.update(frame)
            if self.camera_motion_estimator is not None
            else None
        )
        camera_motion_ms = (time.perf_counter() - camera_started) * 1000.0
        detector_skipped = self._processed_frames % self.detector_interval != 0
        started = time.perf_counter()
        raw_detections = [] if detector_skipped else self.detector.detect(frame, roi=None)
        detections = deduplicate_ball_detections(
            raw_detections,
            iou_threshold=self.duplicate_iou_threshold,
            center_scale=self.duplicate_center_scale,
            minimum_center_px=self.duplicate_center_px,
        )
        detector_ms = 0.0 if detector_skipped else (time.perf_counter() - started) * 1000.0

        tracking_started = time.perf_counter()
        motion_xy = (camera_motion.dx, camera_motion.dy) if camera_motion is not None else (0.0, 0.0)
        tracks = self.tracker.update(detections, width, height, camera_motion=motion_xy)
        tracking_ms = (time.perf_counter() - tracking_started) * 1000.0

        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp_s,
            width=width,
            height=height,
            ball_detections=detections,
            ball_tracks=tracks,
            ball_track=tracks[0] if len(tracks) == 1 else None,
            events=[],
            diagnostics={
                "detector_error": getattr(self.detector, "disabled", False),
                "detector_skipped": detector_skipped,
                "detector_ms": detector_ms,
                "raw_detection_count": len(raw_detections),
                "deduplicated_detection_count": len(detections),
                "duplicate_detections_removed": len(raw_detections) - len(detections),
                "tracking_ms": tracking_ms,
                "camera_motion_ms": camera_motion_ms,
                "camera_motion": (
                    self.camera_motion_estimator.diagnostics
                    if self.camera_motion_estimator is not None
                    else {"enabled": False, "dx": 0.0, "dy": 0.0, "applied": False}
                ),
                "tracker": self.tracker.diagnostics,
                "overlay_warnings": [],
            },
        )
        self._processed_frames += 1
        return result

    def reset(self) -> None:
        self.tracker.reset()
        if self.camera_motion_estimator is not None:
            self.camera_motion_estimator.reset()
        self._processed_frames = 0
