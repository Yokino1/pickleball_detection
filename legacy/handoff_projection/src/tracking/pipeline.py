"""
Per-frame orchestrator for the side-view pickleball analysis pipeline.

    read frame
      → ball_detector.detect(frame, roi)
      → ball_tracker.update(detections)
      → court_observation (from calibration file or detector, if available)
      → court_projector.update(observation)
      → if projector: ball_court_xy = projector.project(track.center)
      → player_detector.detect(frame)  (no-op by default)
      → event_detector.update(track, projector_status)
      → FrameResult
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ball_detector import BallDetector
from .player_detector import PlayerDetector
from .types import (
    BallDetection,
    BallTrack,
    CourtInfo,
    FrameResult,
    PlayerDetection,
)

# Lazy imports for optional court module
try:
    from src.court.layout import CourtLayout      # type: ignore[assignment]
    from src.court.observation import CourtObservation
    from src.court.projector import CourtProjector
    COURT_AVAILABLE = True
except ImportError:
    COURT_AVAILABLE = False


class SideViewPipeline:
    """Owns the per-frame processing loop for side-view analysis.

    Three court modes (in priority order):
    1. **keypoint_tracker** — optical-flow tracks annotated keypoints per frame
    2. **court_observation** — static calibration, same keypoints every frame
    3. **none** — mini-court only, no projection

    Usage::

        pipeline = SideViewPipeline(detector, tracker, event_detector,
                                     court_layout, court_projector,
                                     keypoint_tracker=kt, player_detector=pd)
        for frame in video:
            result = pipeline.process_frame(frame, frame_idx, timestamp_s)
    """

    def __init__(
        self,
        detector: BallDetector,
        tracker,          # SimpleBallTracker
        event_detector,   # EventDetector
        court_layout=None,         # CourtLayout or None
        court_projector=None,      # CourtProjector or None
        court_observation=None,    # CourtObservation or None (static calib)
        keypoint_tracker=None,     # any tracker with is_initialised/update/reset
        player_detector=None,      # PlayerDetector or None
        detector_interval: int = 1,
    ):
        self.detector = detector
        self.tracker = tracker
        self.event_detector = event_detector
        self.court_layout = court_layout
        self.court_projector = court_projector
        self.court_observation = court_observation  # static calibration
        self.keypoint_tracker = keypoint_tracker    # optical flow tracker
        self.player_detector = player_detector
        self.detector_interval = max(1, int(detector_interval))
        self._processed_frames = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _ball_projection_anchor(track: Optional[BallTrack]) -> Optional[tuple[float, float]]:
        """Return the image-space point used for court-plane projection.

        For side-view footage, the ball center can project noticeably outside
        the court while the ball is airborne.  The bottom-center of the box is
        a better proxy for the contact ray with the ground plane, so prefer it
        when available.
        """
        if track is None:
            return None

        if track.bbox is not None and len(track.bbox) >= 4:
            x1, y1, x2, y2 = track.bbox[:4]
            return (float((x1 + x2) / 2.0), float(y2))

        if track.center is not None:
            return (float(track.center[0]), float(track.center[1]))

        return None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: Optional[float] = None,
    ) -> FrameResult:
        """Run the full pipeline on a single frame.

        Args:
            frame: BGR image (H, W, 3).
            frame_index: 0-based frame number.
            timestamp_s: Seconds from video start (or None).

        Returns:
            ``FrameResult`` ready for JSONL serialisation.
        """
        h, w = frame.shape[:2]

        # 1. Ball detection (optionally every N frames)
        roi = getattr(self.tracker, "last_roi", None)
        detector_skipped = (
            self.detector_interval > 1
            and (self._processed_frames % self.detector_interval) != 0
        )
        if detector_skipped:
            detections = []
        else:
            detections = self.detector.detect(frame, roi=roi)

        # 2. Ball tracking
        track: BallTrack = self.tracker.update(detections, w, h)

        # 3. Court observation (from optical-flow tracker, static calib, or none)
        observation: Optional[CourtObservation] = None

        if self.keypoint_tracker is not None and self.keypoint_tracker.is_initialised:
            # Optical flow: track keypoints from reference frame
            observation = self.keypoint_tracker.update(frame)
        elif self.court_observation is not None:
            # Static calibration: reuse same keypoints
            observation = self.court_observation

        # 4. Court projection
        projection_status = "none"
        ball_court_xy = None
        visible_keypoints: list[int] = []

        if self.court_projector is not None and observation is not None:
            projection_status = self.court_projector.update(observation)
            visible_keypoints = observation.visible_indices

            # Project missing sideline keypoints via homography
            if self.court_projector.is_available:
                extra = self.court_projector.project_missing_keypoints(observation)
                if extra:
                    observation.keypoints.update(extra)
                    observation.confidences.update(
                        {k: 0.5 for k in extra}  # lower confidence for projected points
                    )
                    visible_keypoints = observation.visible_indices
        elif self.court_projector is not None:
            projection_status = self.court_projector.status or "unavailable"

        # Project ball to court coords if homography available
        if (
            self.court_projector is not None
            and self.court_projector.is_available
            and track is not None
        ):
            projection_anchor = self._ball_projection_anchor(track)
            if projection_anchor is not None:
                ball_court_xy = self.court_projector.project(projection_anchor)

        # 4. Court info + zone classification
        ball_zone = None
        projection_warnings = []
        if ball_court_xy is not None:
            try:
                from src.court.zones import classify_ball_zone
                zone_info = classify_ball_zone(ball_court_xy)
                ball_zone = zone_info["zone"]
            except ImportError:
                pass

        # Collect projection quality warnings
        proj_quality = (
            self.court_projector.quality
            if self.court_projector else {}
        )
        if proj_quality.get("status") == "rejected":
            projection_warnings.append(
                f"rejected:{proj_quality.get('skip_reason','?')}"
            )
        if proj_quality.get("cache_age", 0) > 0:
            projection_warnings.append(
                f"cache_age:{proj_quality['cache_age']}"
            )

        court_info = CourtInfo(
            layout="pickleball_14pt",
            projection_status=projection_status,
            visible_keypoints=visible_keypoints,
            homography_available=(
                self.court_projector.is_available
                if self.court_projector else False
            ),
            ball_court_xy=list(ball_court_xy) if ball_court_xy else None,
            ball_zone=ball_zone,
            projection_warnings=projection_warnings,
        )

        # 5. Player detection (no-op by default)
        players: list[PlayerDetection] = []
        if self.player_detector is not None:
            players = self.player_detector.detect(frame)

        # 6. Events
        detector_error = getattr(self.detector, "disabled", False)
        events = self.event_detector.update(
            track,
            frame_index=frame_index,
            projection_status=projection_status,
            frame_width=w,
            frame_height=h,
        )

        # 7. Diagnostics
        dropped_reason = getattr(self.tracker, "drop_reason", None)
        diagnostics = {
            "detector_error": detector_error,
            "detector_skipped": detector_skipped,
            "dropped_detection_reason": dropped_reason,
            "overlay_warnings": [],
            # Keypoint tracking info (for overlay rendering)
            "kp_tracked": visible_keypoints,
            "kp_observation": observation.to_dict() if observation else None,
            "kp_tracker_quality": (
                self.keypoint_tracker.quality_report()
                if self.keypoint_tracker and self.keypoint_tracker.is_initialised
                else None
            ),
            # Court projection quality
            "projection_quality": (
                self.court_projector.quality
                if self.court_projector else {}
            ),
        }

        result = FrameResult(
            frame_index=frame_index,
            timestamp=timestamp_s,
            ball_detections=detections,
            ball_track=track,
            court=court_info,
            players=players,
            events=events,
            diagnostics=diagnostics,
            width=w,
            height=h,
        )
        self._processed_frames += 1
        return result

    def reset(self) -> None:
        """Reset all sub-module state (e.g. when seeking or switching video)."""
        self.tracker.reset()
        self.event_detector.reset()
        self._processed_frames = 0
        if self.court_projector is not None:
            self.court_projector.reset()
        if self.keypoint_tracker is not None:
            self.keypoint_tracker.reset()
