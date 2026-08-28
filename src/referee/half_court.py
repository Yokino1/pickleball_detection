"""Single-half serve validation and conservative demo scoring.

This module is deliberately downstream of R9 tracking and court projection.  It
does not change detections, local track association, or the maintained
dual-camera referee.  A half-court camera can validate only the visible
player's service position; an opponent service is inferred from an incoming
observed trajectory and is marked as visually unverified.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from ..court.layout import COURT_LENGTH_FT, NET_Y_FT


WAIT_SERVE = "WAIT_SERVE"
SERVE_CONFIRMING = "SERVE_CONFIRMING"
IN_RALLY = "IN_RALLY"
POST_SCORE_GUARD = "POST_SCORE_GUARD"

VISIBLE = "visible"
OPPONENT = "opponent"
UNKNOWN_SERVER = "unknown"
VALID_HALVES = {"low_y", "high_y"}


@dataclass(frozen=True)
class HalfCourtScoreEvent:
    """One idempotent point decision made from the half-court view."""

    rally_id: int
    frame_index: int
    timestamp_s: float
    winner: str
    reason: str
    score_before: dict[str, int]
    score_after: dict[str, int]
    next_server: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "winner": self.winner,
            "reason": self.reason,
            "score_before": dict(self.score_before),
            "score_after": dict(self.score_after),
            "next_server": self.next_server,
        }


@dataclass(frozen=True)
class HalfCourtFrameResult:
    """Per-frame state rendered on the single-half demo output."""

    phase: str
    frame_index: int
    timestamp_s: float
    rally_id: int | None
    visible_half: str
    server: str
    score: dict[str, int]
    visible_player_behind_baseline: bool
    serve_candidate: str | None
    serve_action_visible: bool | None
    last_event: str | None
    warnings: tuple[str, ...] = ()
    score_event: HalfCourtScoreEvent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "rally_id": self.rally_id,
            "visible_half": self.visible_half,
            "server": self.server,
            "score": dict(self.score),
            "visible_player_behind_baseline": (
                self.visible_player_behind_baseline
            ),
            "serve_candidate": self.serve_candidate,
            "serve_action_visible": self.serve_action_visible,
            "last_event": self.last_event,
            "warnings": list(self.warnings),
            "score_event": (
                self.score_event.to_dict()
                if self.score_event is not None
                else None
            ),
        }


@dataclass(frozen=True)
class _ServeSample:
    frame_index: int
    timestamp_s: float
    court_xy: tuple[float, float]


class HalfCourtServeScoreEngine:
    """Validate half-court serves and apply the requested lost-ball fallback.

    Rules implemented here are intentionally narrow:

    * At boot and after a completed rally, wait for a new serve.
    * A visible-side serve requires a player foot point behind the baseline,
      a hit candidate (or a ball origin behind that baseline), and consecutive
      directly observed ball motion toward the net.
    * An opponent serve cannot validate the unseen player's body position.  It
      is inferred only from consecutive observed ball motion from the net side
      toward the visible baseline and is labelled as unverified.
    * Only an active rally can time out.  If directly observed ball data is
      unavailable for ``lost_timeout_ms``, the opponent receives one point and
      becomes the next server, exactly as requested for the half-court demo.
    """

    def __init__(
        self,
        *,
        visible_half: str,
        lost_timeout_ms: float = 5000.0,
        post_score_guard_ms: float = 800.0,
        serve_confirm_observed_frames: int = 3,
        serve_min_displacement_ft: float = 1.0,
        serve_confirmation_window_ms: float = 800.0,
        baseline_margin_ft: float = 0.15,
        opponent_entry_band_ft: float = 8.0,
        opponent_return_reentry_ms: float = 400.0,
    ) -> None:
        if visible_half not in VALID_HALVES:
            raise ValueError(
                f"visible_half must be one of {sorted(VALID_HALVES)}"
            )
        self.visible_half = visible_half
        self.lost_timeout_s = max(0.0, float(lost_timeout_ms) / 1000.0)
        self.post_score_guard_s = max(
            0.0,
            float(post_score_guard_ms) / 1000.0,
        )
        self.serve_confirm_frames = max(
            2,
            int(serve_confirm_observed_frames),
        )
        self.serve_min_displacement_ft = max(
            0.0,
            float(serve_min_displacement_ft),
        )
        self.serve_window_s = max(
            0.001,
            float(serve_confirmation_window_ms) / 1000.0,
        )
        self.baseline_margin_ft = max(0.0, float(baseline_margin_ft))
        self.opponent_entry_band_ft = max(
            0.0,
            float(opponent_entry_band_ft),
        )
        self.opponent_return_reentry_s = max(
            0.0,
            float(opponent_return_reentry_ms) / 1000.0,
        )

        self.phase = WAIT_SERVE
        self.server = UNKNOWN_SERVER
        self.score = {VISIBLE: 0, OPPONENT: 0}
        self._next_rally_id = 1
        self._active_rally_id: int | None = None
        self._last_observed_time_s: float | None = None
        self._guard_until_s = 0.0
        self._serve_candidate: str | None = None
        self._serve_action_visible: bool | None = None
        self._serve_samples: deque[_ServeSample] = deque(maxlen=32)
        self._last_event: str | None = None
        self._settled_rallies: set[int] = set()
        self._last_hitter: str | None = None
        self._visible_hit_pending = False
        self._visible_ball_departed = False
        self._missing_started_s: float | None = None

    def update(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        ball_court_xy: tuple[float, float] | None,
        ball_observed: bool,
        paddle_hit_candidate: bool,
        player_foot_court_xy: list[tuple[float, float]],
        event_names: list[str] | tuple[str, ...] = (),
    ) -> HalfCourtFrameResult:
        """Advance one video frame without feeding state back into tracking."""
        frame_index = int(frame_index)
        now_s = float(timestamp_s)
        point = self._finite_point(ball_court_xy) if ball_observed else None
        behind_baseline = any(
            self.is_behind_visible_baseline(item)
            for item in player_foot_court_xy
        )
        score_event: HalfCourtScoreEvent | None = None
        warnings: list[str] = []
        event_name_set = {str(value) for value in event_names}

        if self.phase == POST_SCORE_GUARD and now_s >= self._guard_until_s:
            self.phase = WAIT_SERVE
            self._serve_samples.clear()
            self._serve_candidate = None
            self._serve_action_visible = None
            self._last_event = "ready_for_next_serve"

        if self.phase == WAIT_SERVE:
            if point is not None:
                self._append_serve_sample(frame_index, now_s, point)

            visible_candidate = (
                self.server in {UNKNOWN_SERVER, VISIBLE}
                and behind_baseline
                and point is not None
                and (
                    paddle_hit_candidate
                    or self.is_behind_visible_baseline(point)
                )
            )
            if visible_candidate:
                self.phase = SERVE_CONFIRMING
                self._serve_candidate = VISIBLE
                self._serve_action_visible = True
                self._serve_samples.clear()
                self._append_serve_sample(frame_index, now_s, point)
                self._last_event = "visible_serve_candidate"
            elif (
                self.server in {UNKNOWN_SERVER, OPPONENT}
                and self._opponent_serve_confirmed()
            ):
                self._confirm_serve(OPPONENT, now_s)
                self._serve_action_visible = False
                self._last_event = "opponent_serve_inferred"
                warnings.append("opponent_serve_action_not_visible")

        elif self.phase == SERVE_CONFIRMING:
            if self._serve_samples and (
                now_s - self._serve_samples[0].timestamp_s
                > self.serve_window_s
            ):
                self.phase = WAIT_SERVE
                self._serve_samples.clear()
                self._serve_candidate = None
                self._serve_action_visible = None
                self._last_event = "serve_candidate_timeout"
            elif point is not None:
                self._append_serve_sample(frame_index, now_s, point)
                if (
                    self._serve_candidate == VISIBLE
                    and self._visible_serve_confirmed()
                ):
                    self._confirm_serve(VISIBLE, now_s)
                    self._serve_action_visible = True
                    self._last_event = "visible_serve_confirmed"

        elif self.phase == IN_RALLY:
            if "paddle_hit_candidate" in event_name_set:
                self._last_hitter = VISIBLE
                self._visible_hit_pending = True
                self._visible_ball_departed = False
                self._missing_started_s = None
                self._last_event = "visible_hit_confirmed"

            if point is not None:
                if self._visible_ball_departed:
                    # A ball that left the visible half after a visible hit and
                    # later re-entered is treated as an unseen opponent return.
                    self._last_hitter = OPPONENT
                    self._visible_hit_pending = False
                    self._visible_ball_departed = False
                    self._last_event = "opponent_return_inferred"
                self._missing_started_s = None
                self._last_observed_time_s = now_s
            else:
                if self._missing_started_s is None:
                    self._missing_started_s = now_s
                if (
                    self._visible_hit_pending
                    and now_s - self._missing_started_s
                    >= self.opponent_return_reentry_s
                ):
                    self._visible_ball_departed = True

            if "out_of_bounds_bounce_candidate" in event_name_set:
                score_event = self.award_point(
                    VISIBLE,
                    reason="half_court_opponent_ball_out",
                    frame_index=frame_index,
                    timestamp_s=now_s,
                )
                self._last_event = "visible_scored_opponent_ball_out"
            elif "second_bounce_candidate" in event_name_set:
                score_event = self.award_point(
                    OPPONENT,
                    reason="half_court_second_bounce_visible_side",
                    frame_index=frame_index,
                    timestamp_s=now_s,
                )
                self._last_event = "opponent_scored_second_bounce"
            elif (
                "bounce_candidate" in event_name_set
                and self._visible_hit_pending
                and not self._visible_ball_departed
            ):
                score_event = self.award_point(
                    OPPONENT,
                    reason="half_court_visible_ball_failed_to_leave_half",
                    frame_index=frame_index,
                    timestamp_s=now_s,
                )
                self._last_event = "opponent_scored_visible_ball_net"
            elif (
                self.lost_timeout_s > 0.0
                and self._last_observed_time_s is not None
                and now_s - self._last_observed_time_s
                >= self.lost_timeout_s
            ):
                score_event = self.award_point(
                    OPPONENT,
                    reason="half_court_ball_lost_timeout",
                    frame_index=frame_index,
                    timestamp_s=now_s,
                )
                self._last_event = "opponent_scored_ball_lost"

        return HalfCourtFrameResult(
            phase=self.phase,
            frame_index=frame_index,
            timestamp_s=now_s,
            rally_id=self._active_rally_id,
            visible_half=self.visible_half,
            server=self.server,
            score=dict(self.score),
            visible_player_behind_baseline=behind_baseline,
            serve_candidate=self._serve_candidate,
            serve_action_visible=self._serve_action_visible,
            last_event=self._last_event,
            warnings=tuple(warnings),
            score_event=score_event,
        )

    def award_point(
        self,
        winner: str,
        *,
        reason: str,
        frame_index: int,
        timestamp_s: float,
    ) -> HalfCourtScoreEvent | None:
        """Settle the active half-court rally at most once."""
        if winner not in {VISIBLE, OPPONENT}:
            raise ValueError("winner must be 'visible' or 'opponent'")
        rally_id = self._active_rally_id
        if rally_id is None or rally_id in self._settled_rallies:
            return None
        before = dict(self.score)
        self.score[winner] += 1
        self.server = winner
        event = HalfCourtScoreEvent(
            rally_id=rally_id,
            frame_index=int(frame_index),
            timestamp_s=float(timestamp_s),
            winner=winner,
            reason=str(reason),
            score_before=before,
            score_after=dict(self.score),
            next_server=winner,
        )
        self._settled_rallies.add(rally_id)
        self.phase = POST_SCORE_GUARD
        self._guard_until_s = float(timestamp_s) + self.post_score_guard_s
        self._active_rally_id = None
        self._last_observed_time_s = None
        self._serve_samples.clear()
        self._serve_candidate = None
        self._serve_action_visible = None
        self._last_hitter = None
        self._visible_hit_pending = False
        self._visible_ball_departed = False
        self._missing_started_s = None
        return event

    def is_behind_visible_baseline(
        self,
        court_xy: tuple[float, float],
    ) -> bool:
        """Return whether a ground point is beyond this half's baseline."""
        point = self._finite_point(court_xy)
        if point is None:
            return False
        _x_ft, y_ft = point
        if self.visible_half == "low_y":
            return y_ft < -self.baseline_margin_ft
        return y_ft > COURT_LENGTH_FT + self.baseline_margin_ft

    def diagnostics(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "visible_half": self.visible_half,
            "server": self.server,
            "score": dict(self.score),
            "active_rally_id": self._active_rally_id,
            "settled_rallies": sorted(self._settled_rallies),
            "lost_timeout_ms": self.lost_timeout_s * 1000.0,
            "serve_confirm_observed_frames": self.serve_confirm_frames,
            "serve_min_displacement_ft": self.serve_min_displacement_ft,
            "last_hitter": self._last_hitter,
        }

    def _append_serve_sample(
        self,
        frame_index: int,
        timestamp_s: float,
        point: tuple[float, float],
    ) -> None:
        if self._serve_samples and (
            timestamp_s - self._serve_samples[-1].timestamp_s
            > self.serve_window_s
        ):
            self._serve_samples.clear()
        self._serve_samples.append(
            _ServeSample(frame_index, timestamp_s, point)
        )
        while (
            self._serve_samples
            and timestamp_s - self._serve_samples[0].timestamp_s
            > self.serve_window_s
        ):
            self._serve_samples.popleft()

    def _visible_serve_confirmed(self) -> bool:
        if len(self._serve_samples) < self.serve_confirm_frames:
            return False
        samples = list(self._serve_samples)[-self.serve_confirm_frames :]
        start_y = samples[0].court_xy[1]
        end_y = samples[-1].court_xy[1]
        displacement = self._netward_displacement(start_y, end_y)
        return displacement >= self.serve_min_displacement_ft

    def _opponent_serve_confirmed(self) -> bool:
        if len(self._serve_samples) < self.serve_confirm_frames:
            return False
        samples = list(self._serve_samples)[-self.serve_confirm_frames :]
        start_y = samples[0].court_xy[1]
        end_y = samples[-1].court_xy[1]
        if abs(start_y - NET_Y_FT) > self.opponent_entry_band_ft:
            return False
        displacement = -self._netward_displacement(start_y, end_y)
        return displacement >= self.serve_min_displacement_ft

    def _netward_displacement(self, start_y: float, end_y: float) -> float:
        if self.visible_half == "low_y":
            return end_y - start_y
        return start_y - end_y

    def _confirm_serve(self, server: str, timestamp_s: float) -> None:
        self.server = server
        self.phase = IN_RALLY
        self._active_rally_id = self._next_rally_id
        self._next_rally_id += 1
        self._last_observed_time_s = float(timestamp_s)
        self._serve_candidate = None
        self._serve_samples.clear()
        self._last_hitter = server
        self._visible_hit_pending = server == VISIBLE
        self._visible_ball_departed = False
        self._missing_started_s = None

    @staticmethod
    def _finite_point(
        value: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if value is None or len(value) < 2:
            return None
        x_ft = float(value[0])
        y_ft = float(value[1])
        if not (-1e12 < x_ft < 1e12 and -1e12 < y_ft < 1e12):
            return None
        return x_ft, y_ft


def court_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Small public helper used by display/tests without NumPy."""
    return hypot(second[0] - first[0], second[1] - first[1])
