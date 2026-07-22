"""Detection-only ball tracking pipeline, independent of court projection."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .ball_detector import BallDetector
from .multi_ball_tracker import MultiBallTracker
from .types import FrameResult


class BallTrackingPipeline:
    """Run full-frame detection and multi-ball tracking for each video frame."""

    def __init__(self, detector: BallDetector, tracker: MultiBallTracker, detector_interval: int = 1):
        self.detector = detector
        self.tracker = tracker
        self.detector_interval = max(1, int(detector_interval))
        self._processed_frames = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: Optional[float] = None,
    ) -> FrameResult:
        height, width = frame.shape[:2]
        detector_skipped = self._processed_frames % self.detector_interval != 0
        started = time.perf_counter()
        detections = [] if detector_skipped else self.detector.detect(frame, roi=None)
        detector_ms = 0.0 if detector_skipped else (time.perf_counter() - started) * 1000.0

        tracking_started = time.perf_counter()
        tracks = self.tracker.update(detections, width, height)
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
                "tracking_ms": tracking_ms,
                "tracker": self.tracker.diagnostics,
                "overlay_warnings": [],
            },
        )
        self._processed_frames += 1
        return result

    def reset(self) -> None:
        self.tracker.reset()
        self._processed_frames = 0
