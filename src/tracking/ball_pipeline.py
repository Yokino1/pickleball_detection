"""Detection-only ball tracking pipeline, independent of court projection."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .ball_detector import BallDetector
from .camera_motion import CameraMotionEstimator
from .fast_motion import FastMotionProposalGenerator
from .multi_ball_tracker import MultiBallTracker
from .person_detector import PersonDetector
from .person_tracking import PersonBoxTracker, PlayerSelector
from .temporal_motion import TemporalMotionFilter
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
        temporal_motion_filter: TemporalMotionFilter | None = None,
        fast_motion_proposal_generator: FastMotionProposalGenerator | None = None,
        person_detector: PersonDetector | None = None,
        person_tracker: PersonBoxTracker | None = None,
        player_selector: PlayerSelector | None = None,
        person_detector_interval: int = 5,
    ):
        self.detector = detector
        self.tracker = tracker
        self.detector_interval = max(1, int(detector_interval))
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.duplicate_center_scale = float(duplicate_center_scale)
        self.duplicate_center_px = float(duplicate_center_px)
        self.camera_motion_estimator = camera_motion_estimator
        self.temporal_motion_filter = temporal_motion_filter
        self.fast_motion_proposal_generator = fast_motion_proposal_generator
        self.person_detector = person_detector
        self.person_tracker = person_tracker
        self.player_selector = player_selector
        self.person_detector_interval = max(1, int(person_detector_interval))
        self._processed_frames = 0
        self._last_tracks = []

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: Optional[float] = None,
        search_roi: Optional[list[float]] = None,
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
        roi_retry_used = False
        if not detector_skipped and not raw_detections and search_roi is not None:
            raw_detections = self.detector.detect(frame, roi=search_roi)
            roi_retry_used = True
        deduplicated_detections = deduplicate_ball_detections(
            raw_detections,
            iou_threshold=self.duplicate_iou_threshold,
            center_scale=self.duplicate_center_scale,
            minimum_center_px=self.duplicate_center_px,
        )
        detector_ms = (
            0.0
            if detector_skipped
            else (time.perf_counter() - started) * 1000.0
        )
        motion_xy = (camera_motion.dx, camera_motion.dy) if camera_motion is not None else (0.0, 0.0)
        fast_motion_roi = None
        if not deduplicated_detections:
            fast_motion_roi = search_roi or self._fast_track_search_roi(width, height)
        fast_motion_started = time.perf_counter()
        fast_motion_proposals = (
            self.fast_motion_proposal_generator.propose(
                frame,
                timestamp_s=timestamp_s,
                search_roi=fast_motion_roi,
                camera_motion=motion_xy,
            )
            if self.fast_motion_proposal_generator is not None
            else []
        )
        fast_motion_ms = (
            (time.perf_counter() - fast_motion_started) * 1000.0
            if self.fast_motion_proposal_generator is not None
            else 0.0
        )
        detections = deduplicate_ball_detections(
            [*deduplicated_detections, *fast_motion_proposals],
            iou_threshold=self.duplicate_iou_threshold,
            center_scale=self.duplicate_center_scale,
            minimum_center_px=self.duplicate_center_px,
        )
        temporal_started = time.perf_counter()
        if self.temporal_motion_filter is not None:
            detections = self.temporal_motion_filter.filter(
                frame,
                detections,
                camera_motion=motion_xy,
            )
        temporal_motion_ms = (
            (time.perf_counter() - temporal_started) * 1000.0
            if self.temporal_motion_filter is not None
            else 0.0
        )
        person_detector_skipped = (
            self.person_detector is None
            or self._processed_frames % self.person_detector_interval != 0
        )
        person_started = time.perf_counter()
        raw_people = (
            None
            if person_detector_skipped
            else self.person_detector.detect(frame)
        )
        person_detector_ms = (
            0.0
            if person_detector_skipped
            else (time.perf_counter() - person_started) * 1000.0
        )
        people = (
            self.person_tracker.update(
                raw_people,
                width,
                height,
                timestamp_s=timestamp_s,
            )
            if self.person_tracker is not None
            else (raw_people or [])
        )
        if self.player_selector is not None:
            people = self.player_selector.select(people, width, height)
        contact_zones = [
            list(person.bbox)
            for person in people
            if person.eligible_player
        ]

        tracking_started = time.perf_counter()
        tracks = self.tracker.update(
            detections,
            width,
            height,
            camera_motion=motion_xy,
            timestamp_s=timestamp_s,
            contact_zones=contact_zones,
        )
        self._last_tracks = list(tracks)
        tracking_ms = (time.perf_counter() - tracking_started) * 1000.0

        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp_s,
            width=width,
            height=height,
            ball_detections=detections,
            ball_tracks=tracks,
            ball_track=tracks[0] if len(tracks) == 1 else None,
            players=people,
            events=[],
            diagnostics={
                "detector_error": getattr(self.detector, "disabled", False),
                "detector_skipped": detector_skipped,
                "search_roi": search_roi,
                "roi_retry_used": roi_retry_used,
                "detector_ms": detector_ms,
                "raw_detection_count": len(raw_detections),
                "deduplicated_detection_count": len(deduplicated_detections),
                "duplicate_detections_removed": (
                    len(raw_detections) - len(deduplicated_detections)
                ),
                "temporal_motion_ms": temporal_motion_ms,
                "temporal_motion": (
                    self.temporal_motion_filter.diagnostics
                    if self.temporal_motion_filter is not None
                    else {"enabled": False}
                ),
                "fast_motion_ms": fast_motion_ms,
                "fast_motion_proposal_count": len(fast_motion_proposals),
                "fast_motion": (
                    self.fast_motion_proposal_generator.diagnostics
                    if self.fast_motion_proposal_generator is not None
                    else {"enabled": False}
                ),
                "tracking_ms": tracking_ms,
                "person_detector_enabled": self.person_detector is not None,
                "person_detector_error": (
                    getattr(self.person_detector, "disabled", False)
                    if self.person_detector is not None
                    else False
                ),
                "person_detector_skipped": person_detector_skipped,
                "person_detector_ms": person_detector_ms,
                "person_detection_count": len(raw_people or []),
                "person_track_count": len(people),
                "eligible_player_count": len(contact_zones),
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
        if self.temporal_motion_filter is not None:
            self.temporal_motion_filter.reset()
        if self.fast_motion_proposal_generator is not None:
            self.fast_motion_proposal_generator.reset()
        if self.person_tracker is not None:
            self.person_tracker.reset()
        self._processed_frames = 0
        self._last_tracks = []

    def _fast_track_search_roi(
        self,
        frame_width: int,
        frame_height: int,
    ) -> Optional[list[float]]:
        if self.fast_motion_proposal_generator is None:
            return None
        frame_scale = self.fast_motion_proposal_generator.frame_scale(
            frame_width
        )
        minimum_speed = (
            self.fast_motion_proposal_generator.min_speed_px_per_second
            * frame_scale
        )
        fast_tracks = [
            track
            for track in self._last_tracks
            if track.center is not None
            and track.velocity is not None
            and float(np.hypot(*track.velocity)) >= minimum_speed
        ]
        if not fast_tracks:
            return None
        track = max(
            fast_tracks,
            key=lambda item: float(np.hypot(*item.velocity)),
        )
        if track.roi is not None:
            return list(track.roi)
        radius = max(24.0, frame_width * 0.10)
        return [
            max(0.0, float(track.center[0]) - radius),
            max(0.0, float(track.center[1]) - radius),
            min(float(frame_width), float(track.center[0]) + radius),
            min(float(frame_height), float(track.center[1]) + radius),
        ]
