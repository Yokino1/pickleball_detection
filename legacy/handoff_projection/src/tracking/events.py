"""
Simple candidate-event detector.

All events are **candidates** — weak hypotheses, not strong conclusions.
No velocity-based bounce detection, no in/out calls, no complex state machines.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional


class EventDetector:
    """Emit candidate events from tracker + projector state.

    All events are candidates.  The caller is responsible for recording
    them in JSONL / overlay; later stages (or human reviewers) can
    confirm or reject them.
    """

    def __init__(
        self,
        ball_lost_timeout_frames: int = 5,
        leave_view_margin_px: float = 12.0,
        bounce_window: int = 11,       # must be odd; ≥5 points before/after peak
        min_bounce_gap: int = 15,      # frames between bounce candidates
    ):
        self.ball_lost_timeout = ball_lost_timeout_frames
        self.leave_view_margin = leave_view_margin_px
        self.bounce_window = bounce_window
        self.min_bounce_gap = min_bounce_gap

        self._prev_track: Any = None
        self._consecutive_predicted: int = 0
        self._last_bounce_frame: int = -999

        # Y-trajectory buffer for bounce detection
        self._y_history: deque[float] = deque(maxlen=bounce_window)
        self._frame_history: deque[int] = deque(maxlen=bounce_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        track: Any = None,
        frame_index: int = 0,
        projection_status: str = "none",
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> list[str]:
        """Return candidate events for this frame.

        *track* is a ``BallTrack`` or any object with a ``.status``
        attribute (``"observed"`` | ``"predicted"`` | ``"absent"``).

        Returns:
            List of event strings.
        """
        events: list[str] = []

        if track is None:
            events.append("ball_missing")
            self._prev_track = None
            self._consecutive_predicted = 0
            return events

        # Resolve status from BallTrack.status
        status = self._resolve_status(track)

        # --- observed --------------------------------------------------
        if status == "observed":
            events.append("ball_seen")
            self._consecutive_predicted = 0

            # Track Y for bounce detection
            center = self._resolve_center(track)
            if center is not None:
                self._y_history.append(center[1])
                self._frame_history.append(frame_index)
                if self._check_bounce(frame_index):
                    events.append("bounce_candidate")
                    self._last_bounce_frame = frame_index

        # --- predicted -------------------------------------------------
        elif status == "predicted":
            self._consecutive_predicted += 1
            if self._consecutive_predicted > self.ball_lost_timeout:
                events.append("ball_lost")

        # --- absent ----------------------------------------------------
        else:  # status == "absent"
            was_active = self._prev_track is not None and (
                self._resolve_status(self._prev_track) in ("observed", "predicted")
            )
            if was_active:
                events.append("ball_lost")
                leave = self._detect_leave_view(track, frame_width, frame_height)
                if leave is not None:
                    events.append(leave)
            self._consecutive_predicted = 0

        # --- projection -------------------------------------------------
        if projection_status not in ("homography", "cached"):
            events.append("projection_unavailable")

        self._prev_track = track
        return events

    def reset(self) -> None:
        self._prev_track = None
        self._consecutive_predicted = 0
        self._last_bounce_frame = -999
        self._y_history.clear()
        self._frame_history.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_status(track: Any) -> str:
        """Return 'observed' | 'predicted' | 'absent' from track status."""
        return getattr(track, "status", "absent")

    @staticmethod
    def _resolve_center(track: Any) -> Optional[list[float]]:
        """Return [cx, cy] or None."""
        center = getattr(track, "center", None)
        return center

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_bounce(self, frame_index: int) -> bool:
        """Weak bounce candidate: Y trajectory has a local maximum."""
        if len(self._y_history) < self.bounce_window:
            return False
        if frame_index - self._last_bounce_frame < self.min_bounce_gap:
            return False

        ys = list(self._y_history)
        mid = len(ys) // 2
        mid_y = ys[mid]
        before = ys[:mid]
        after = ys[mid + 1:]

        if not before or not after:
            return False

        # Local maximum in image Y: ball goes down (Y increases),
        # bounces, then goes up (Y decreases)
        if mid_y > max(before) and mid_y >= max(after):
            if all(y > 0 for y in ys):
                return True

        return False

    def _detect_leave_view(
        self,
        track: Any,
        fw: int,
        fh: int,
    ) -> Optional[str]:
        """Detect which edge the ball left from."""
        center = self._resolve_center(track)
        if center is None:
            return None
        cx, cy = center[0], center[1]
        m = self.leave_view_margin
        if cx <= m:
            return "leave_view_left"
        if cx >= fw - m:
            return "leave_view_right"
        if cy <= m:
            return "leave_view_top"
        if cy >= fh - m:
            return "leave_view_bottom"
        return None
