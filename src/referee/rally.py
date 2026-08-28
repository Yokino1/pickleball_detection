"""Read-only rally, net-fault and score inference after court projection."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

from ..court.layout import CourtLayout
from ..court.projector import ProjectionResult
from .compat_0811 import (
    map_event_type_0811,
    rally_over_0811,
    target_ball_state_0811,
)
from .scoring import ScoreRecorder
from .types import HitRecord, RallyResult, RefereeFrameResult

LOW_Y_HALF = "low_y"
HIGH_Y_HALF = "high_y"
VALID_HALVES = {LOW_Y_HALF, HIGH_Y_HALF}

WAIT_SERVE = "WAIT_SERVE"
SERVE_CONFIRMING = "SERVE_CONFIRMING"
IN_RALLY = "IN_RALLY"
POST_RALLY_GUARD = "POST_RALLY_GUARD"


@dataclass(frozen=True)
class _ObservedSample:
    timestamp_s: float
    frame_index: int
    court_xy: tuple[float, float]
    half: str


@dataclass(frozen=True)
class _PendingBounce:
    """One ambiguous contact kept inside the active rally for later evidence."""

    timestamp_s: float
    frame_index: int
    contact_half: str | None
    contact_xy: tuple[float, float] | None
    contact_inside: bool | None
    reason: str
    evidence: tuple[str, ...]
    quality_warnings: tuple[str, ...]


@dataclass
class _HitRecordState:
    """Mutable aggregation state; public output is an immutable HitRecord."""

    hit_count: int
    hit_half: str
    hit_side: str
    route_destination_half: str
    route_destination_side: str
    hit_start_frame_index: int
    hit_start_timestamp_s: float
    hit_end_frame_index: int | None = None
    hit_end_timestamp_s: float | None = None
    event_type_0811: str = "NONE"
    first_landing_frame_index: int | None = None
    first_landing_timestamp_s: float | None = None
    first_landing_position: tuple[float, float] | None = None
    first_landing_inout: str | None = None
    decision_latency_ms: float | None = None

    def snapshot(self) -> HitRecord:
        return HitRecord(
            hit_count=self.hit_count,
            hit_half=self.hit_half,
            hit_side=self.hit_side,
            route_destination_half=self.route_destination_half,
            route_destination_side=self.route_destination_side,
            hit_start_frame_index=self.hit_start_frame_index,
            hit_start_timestamp_s=self.hit_start_timestamp_s,
            hit_end_frame_index=self.hit_end_frame_index,
            hit_end_timestamp_s=self.hit_end_timestamp_s,
            event_type_0811=self.event_type_0811,
            first_landing_frame_index=self.first_landing_frame_index,
            first_landing_timestamp_s=self.first_landing_timestamp_s,
            first_landing_position=self.first_landing_position,
            first_landing_inout=self.first_landing_inout,
            decision_latency_ms=self.decision_latency_ms,
        )


class RallyRefereeEngine:
    """Infer rally outcomes without changing R9 or its event candidates.

    Only valid, directly observed court projections enter motion confirmation.
    A hit candidate is retained as supporting evidence, but can never start a
    serve, confirm a net crossing, or create a new route by itself.
    """

    def __init__(
        self,
        layout: CourtLayout | None = None,
        *,
        enabled: bool = True,
        low_y_side: str = "left",
        high_y_side: str = "right",
        scoring_mode: str = "rally_point",
        initial_score: dict[str, int] | None = None,
        net_deadband_ft: float = 0.75,
        route_confirm_observed_frames: int = 3,
        route_min_netward_displacement_ft: float = 1.0,
        net_cross_confirm_observed_frames: int = 2,
        net_cross_min_displacement_ft: float = 0.15,
        max_observation_gap_ms: float = 120.0,
        unavailable_timeout_ms: float = 2000.0,
        post_rally_guard_ms: float = 800.0,
        post_rally_rearm_missing_ms: float = 200.0,
        post_rally_rearm_stable_observed_frames: int = 5,
        post_rally_rearm_stable_displacement_ft: float = 0.5,
        hit_assist_window_ms: float = 300.0,
        observation_margin_ft: float = 6.0,
        motion_confirmation_window_ms: float | None = None,
        pending_bounce_timeout_ms: float | None = None,
        require_serve_arming: bool = True,
        expected_server_side: str | None = None,
        service_ownership_enabled: bool = False,
        initial_server_side: str | None = None,
        require_server_player_center_outside_court: bool = False,
    ) -> None:
        if low_y_side == high_y_side:
            raise ValueError("low_y_side and high_y_side must be distinct")
        self.layout = layout or CourtLayout()
        self.enabled = bool(enabled)
        self.half_to_side = {
            LOW_Y_HALF: str(low_y_side),
            HIGH_Y_HALF: str(high_y_side),
        }
        self.side_to_half = {
            side: half for half, side in self.half_to_side.items()
        }
        self.score_recorder = ScoreRecorder(
            teams=(str(low_y_side), str(high_y_side)),
            scoring_mode=scoring_mode,
            initial_score=initial_score,
        )
        self.net_deadband_ft = max(0.0, float(net_deadband_ft))
        self.route_confirm_frames = max(2, int(route_confirm_observed_frames))
        self.route_min_displacement_ft = max(0.0, float(route_min_netward_displacement_ft))
        self.cross_confirm_frames = max(2, int(net_cross_confirm_observed_frames))
        self.cross_min_displacement_ft = max(0.0, float(net_cross_min_displacement_ft))
        self.max_observation_gap_s = max(0.0, float(max_observation_gap_ms) / 1000.0)
        derived_confirmation_window_s = self.max_observation_gap_s * max(
            self.route_confirm_frames - 1,
            self.cross_confirm_frames - 1,
            1,
        )
        self.motion_confirmation_window_s = max(
            0.0,
            (
                derived_confirmation_window_s
                if motion_confirmation_window_ms is None
                else float(motion_confirmation_window_ms) / 1000.0
            ),
        )
        self.unavailable_timeout_s = max(0.0, float(unavailable_timeout_ms) / 1000.0)
        self.pending_bounce_timeout_s = max(
            0.0,
            (
                self.unavailable_timeout_s
                if pending_bounce_timeout_ms is None
                else float(pending_bounce_timeout_ms) / 1000.0
            ),
        )
        self.post_rally_guard_s = max(0.0, float(post_rally_guard_ms) / 1000.0)
        self.post_rally_rearm_missing_s = max(
            0.0,
            float(post_rally_rearm_missing_ms) / 1000.0,
        )
        self.post_rally_rearm_stable_frames = max(
            2,
            int(post_rally_rearm_stable_observed_frames),
        )
        self.post_rally_rearm_stable_displacement_ft = max(
            0.0,
            float(post_rally_rearm_stable_displacement_ft),
        )
        self.hit_assist_window_s = max(0.0, float(hit_assist_window_ms) / 1000.0)
        self.observation_margin_ft = max(0.0, float(observation_margin_ft))
        self.require_serve_arming = bool(require_serve_arming)
        self.expected_server_side: str | None = None
        self.set_expected_server_side(expected_server_side)
        self.service_ownership_enabled = bool(service_ownership_enabled)
        self.require_server_player_center_outside_court = bool(
            require_server_player_center_outside_court
        )
        self.service_owner_side: str | None = None
        self.service_owner_source: str | None = None
        if initial_server_side is not None:
            normalized_initial_server = str(initial_server_side)
            if normalized_initial_server not in self.score_recorder.teams:
                raise ValueError(
                    "initial_server_side must match one configured downstream "
                    f"side; got {normalized_initial_server!r}, expected one of "
                    f"{self.score_recorder.teams!r}"
                )
            if self.service_ownership_enabled:
                self.service_owner_side = normalized_initial_server
                self.service_owner_source = "initial_config"
        # Motion confirmation is bounded by time rather than by the last exact
        # N observations.  The generous hard cap is only a memory guard; stale
        # samples are removed by ``_append_contiguous``.
        history_size = 256
        self._serve_samples: deque[_ObservedSample] = deque(maxlen=history_size)
        self._cross_samples: deque[_ObservedSample] = deque(maxlen=history_size)
        self._return_samples: deque[_ObservedSample] = deque(maxlen=history_size)
        self._late_cross_samples: deque[_ObservedSample] = deque(maxlen=history_size)
        self._guard_stable_samples: deque[_ObservedSample] = deque(
            maxlen=self.post_rally_rearm_stable_frames
        )
        self._serve_arming_stable_samples: deque[_ObservedSample] = deque(
            maxlen=self.post_rally_rearm_stable_frames
        )
        self._waiting_return_samples: deque[_ObservedSample] = deque(
            maxlen=history_size
        )
        self._event_fingerprints: deque[tuple[Any, ...]] = deque(maxlen=64)
        self._event_fingerprint_set: set[tuple[Any, ...]] = set()

        self.phase = WAIT_SERVE
        self._rally_sequence = 0
        self._current_rally_id: int | None = None
        self._rally_start_frame = -1
        self._rally_start_time_s = 0.0
        self._server_half: str | None = None
        self._server_player_centers_court_xy: tuple[
            tuple[float, float], ...
        ] = ()
        self._server_player_center_outside_court: bool | None = None
        self._route_origin_half: str | None = None
        self._route_destination_half: str | None = None
        self._leg_index = 0
        self._net_crossed = False
        self._first_bounce_half: str | None = None
        self._hit_records: list[_HitRecordState] = []
        self._current_hit_record: _HitRecordState | None = None
        self._pending_bounce: _PendingBounce | None = None
        self._last_observed_time_s = float("-inf")
        self._guard_until_s = float("-inf")
        self._guard_rearmed = False
        self._serve_armed = not self.require_serve_arming
        self._serve_arming_missing_since_s: float | None = None
        self._last_hit_candidate_time_s = float("-inf")
        self._waiting_return_hit_sample: _ObservedSample | None = None
        self._last_timestamp_s = float("-inf")
        self._frame_decision_latency_ms: dict[str, float] = {}
        self._frame_target_ball_state_0811 = "target_ball_lost"
        self._frame_eligible_player_centers_court_xy: list[
            tuple[float, float]
        ] = []
        self._frame_eligible_player_center_outside_court = False
        self._decision_latency_stats: dict[str, dict[str, float | int]] = {}

        self._updates = 0
        self._observed_samples = 0
        self._serves_confirmed = 0
        self._routes_confirmed = 0
        self._crossings_confirmed = 0
        self._demo_inferred_results = 0
        self._unknown_results = 0
        self._duplicate_event_candidates = 0
        self._serve_arming_stable_count = 0
        self._serve_arming_missing_count = 0
        self._service_owner_changes = 0
        self._wrong_server_rejections = 0
        self._server_position_rejections = 0
        self._terminal_events: Counter[str] = Counter()

    def set_expected_server_side(self, side: str | None) -> None:
        """Set downstream's expected server for diagnostics only.

        This value never gates serve confirmation and never changes scoring.
        """
        if side is None:
            self.expected_server_side = None
            return
        normalized = str(side)
        if normalized not in self.score_recorder.teams:
            raise ValueError(
                "expected_server_side must match one configured downstream side; "
                f"got {normalized!r}, expected one of {self.score_recorder.teams!r}"
            )
        self.expected_server_side = normalized

    def _set_service_owner(self, side: str, *, source: str) -> None:
        """Persist the side authorized to start the next rally."""
        normalized = str(side)
        if normalized not in self.score_recorder.teams:
            raise ValueError(
                "service owner must match one configured downstream side; "
                f"got {normalized!r}, expected one of "
                f"{self.score_recorder.teams!r}"
            )
        if self.service_owner_side != normalized:
            self._service_owner_changes += 1
        self.service_owner_side = normalized
        self.service_owner_source = str(source)

    def _service_owner_half(self) -> str | None:
        if self.service_owner_side is None:
            return None
        return self.side_to_half[self.service_owner_side]

    @staticmethod
    def _finite_player_centers(values) -> list[tuple[float, float]]:
        centers: list[tuple[float, float]] = []
        for value in values or ():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            try:
                point = (float(value[0]), float(value[1]))
            except (TypeError, ValueError, OverflowError):
                continue
            if all(isfinite(item) for item in point):
                centers.append(point)
        return centers

    def update(
        self,
        projection: ProjectionResult,
        *,
        timestamp_s: float,
        frame_index: int,
        discontinuity_reason: str | None = None,
        eligible_player_centers_court_xy=(),
    ) -> RefereeFrameResult:
        """Consume one projected global-ball frame and return a snapshot."""
        now_s = float(timestamp_s)
        current_frame = int(frame_index)
        if not isfinite(now_s) or now_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if current_frame < 0:
            raise ValueError("frame_index must be non-negative")
        self._updates += 1
        self._frame_decision_latency_ms = {}
        self._frame_target_ball_state_0811 = target_ball_state_0811(projection)
        self._frame_eligible_player_centers_court_xy = (
            self._finite_player_centers(eligible_player_centers_court_xy)
        )
        self._frame_eligible_player_center_outside_court = any(
            not self.layout.contains(point)
            for point in self._frame_eligible_player_centers_court_xy
        )
        frame_events: list[str] = []
        frame_evidence: list[str] = []
        frame_warnings: list[str] = []
        rally_result: RallyResult | None = None

        if now_s < self._last_timestamp_s:
            self._clear_motion_buffers()
            frame_warnings.append("non_monotonic_timestamp_motion_reset")
        self._last_timestamp_s = now_s

        if not self.enabled:
            return self._snapshot(
                current_frame,
                now_s,
                physical_half=None,
                events=frame_events,
                evidence=frame_evidence,
                warnings=["referee_disabled"],
                rally_result=None,
            )

        pre_discontinuity_serve_samples = tuple(self._serve_samples)
        if discontinuity_reason is not None:
            self._clear_motion_buffers()
            frame_warnings.append(f"motion_confirmation_reset:{discontinuity_reason}")

        event = projection.event or {}
        event_names = {str(name) for name in event.get("events", []) if name is not None}
        if "paddle_hit_candidate" in event_names:
            self._last_hit_candidate_time_s = now_s
            frame_evidence.append("paddle_hit_candidate_support_only")

        direct_observed = self._is_direct_observed_projection(projection)
        sample = self._observed_sample(projection, now_s, current_frame)
        physical_half = sample.half if sample is not None else None
        if (
            sample is not None
            and "paddle_hit_candidate" in event_names
            and self.phase in {WAIT_SERVE, SERVE_CONFIRMING}
        ):
            inbound_samples = self._serve_samples
            if (
                discontinuity_reason == "local_track_change"
                and pre_discontinuity_serve_samples
            ):
                # A local tracker ID can legitimately change at paddle impact.
                # Preserve only the already-observed inbound suffix when the
                # same frame also carries an independent hit candidate; all
                # other discontinuities continue to discard motion history.
                inbound_samples = deque(pre_discontinuity_serve_samples)
            inbound_window = self._directional_confirmation_window(
                inbound_samples,
                sample.half,
                toward_net=False,
                required_frames=self.route_confirm_frames,
                min_displacement_ft=self.route_min_displacement_ft,
            )
            if inbound_window:
                self._waiting_return_hit_sample = sample
                self._waiting_return_samples.clear()
                self._waiting_return_samples.append(sample)
                frame_evidence.append(
                    "observed_inbound_route_before_paddle_hit"
                )
        if direct_observed:
            self._last_observed_time_s = now_s
        if sample is not None:
            self._observed_samples += 1

        if self.phase == POST_RALLY_GUARD:
            self._update_guard_rearm(
                sample,
                direct_observed=direct_observed,
                timestamp_s=now_s,
                frame_evidence=frame_evidence,
            )
            if now_s >= self._guard_until_s and self._guard_rearmed:
                self.phase = WAIT_SERVE
                self._serve_armed = True
                self._serve_arming_missing_since_s = None
                self._serve_arming_stable_samples.clear()
                self._serve_samples.clear()
                self._clear_current_rally()
                frame_events.append("ready_for_next_serve")
            else:
                return self._snapshot(
                    current_frame,
                    now_s,
                    physical_half=physical_half,
                    events=frame_events,
                    evidence=frame_evidence,
                    warnings=frame_warnings,
                    rally_result=None,
                )

        if (
            self.phase == WAIT_SERVE
            and self.require_serve_arming
            and not self._serve_armed
        ):
            arming_reason = self._update_serve_arming(
                sample,
                direct_observed=direct_observed,
                predicted=bool(projection.predicted),
                timestamp_s=now_s,
            )
            if arming_reason is not None:
                frame_events.append("serve_armed")
                frame_evidence.append(arming_reason)
        elif self.phase in {WAIT_SERVE, SERVE_CONFIRMING}:
            if sample is not None:
                recovered_return = False
                waiting_return_hit = self._waiting_return_hit_sample
                if waiting_return_hit is not None:
                    if (
                        sample.half != waiting_return_hit.half
                        or sample.timestamp_s - waiting_return_hit.timestamp_s
                        > self.motion_confirmation_window_s
                    ):
                        self._waiting_return_hit_sample = None
                        self._waiting_return_samples.clear()
                    else:
                        if sample.frame_index != waiting_return_hit.frame_index:
                            self._append_contiguous(
                                self._waiting_return_samples,
                                sample,
                                window_s=self.motion_confirmation_window_s,
                            )
                        recovered_window = self._directional_confirmation_window(
                            self._waiting_return_samples,
                            waiting_return_hit.half,
                            toward_net=True,
                            required_frames=self.route_confirm_frames,
                            min_displacement_ft=self.route_min_displacement_ft,
                        )
                        if recovered_window:
                            origin = waiting_return_hit.half
                            server_half = self._service_owner_half() or origin
                            self._start_rally(
                                server_half,
                                waiting_return_hit.frame_index,
                                waiting_return_hit.timestamp_s,
                                route_origin_half=origin,
                                leg_index=2,
                                confirmed_serve=False,
                            )
                            frame_events.append(
                                "rally_recovered_from_observed_return"
                            )
                            frame_events.append("return_route_confirmed")
                            frame_evidence.extend(
                                [
                                    "paddle_hit_candidate",
                                    "observed_inbound_route_before_paddle_hit",
                                    "netward_direction_reversal_confirmed",
                                    "serve_was_not_directly_confirmed",
                                ]
                            )
                            frame_warnings.append(
                                "rally_started_from_mid_rally_return_recovery"
                            )
                            recovered_return = True
                if recovered_return:
                    pass
                else:
                    self._append_contiguous(
                        self._serve_samples,
                        sample,
                        window_s=self.motion_confirmation_window_s,
                    )
                    self.phase = SERVE_CONFIRMING
                    origin = sample.half
                    confirmation_window = self._directional_confirmation_window(
                        self._serve_samples,
                        origin,
                        toward_net=True,
                        required_frames=self.route_confirm_frames,
                        min_displacement_ft=self.route_min_displacement_ft,
                    )
                if not recovered_return and confirmation_window:
                    detected_server_side = self.half_to_side[origin]
                    if (
                        self.service_ownership_enabled
                        and self.service_owner_side is not None
                        and detected_server_side != self.service_owner_side
                    ):
                        self._wrong_server_rejections += 1
                        self._serve_samples.clear()
                        self.phase = WAIT_SERVE
                        frame_events.append(
                            "serve_rejected_wrong_service_owner"
                        )
                        frame_warnings.append(
                            "detected_server_side_does_not_own_service"
                        )
                    elif (
                        self.require_server_player_center_outside_court
                        and not self._frame_eligible_player_center_outside_court
                    ):
                        self._server_position_rejections += 1
                        self._serve_samples.clear()
                        self.phase = WAIT_SERVE
                        frame_events.append(
                            "serve_rejected_player_center_inside_court"
                        )
                        frame_warnings.append(
                            "eligible_server_player_center_not_outside_court"
                        )
                    else:
                        first_sample = confirmation_window[0]
                        if (
                            self.service_ownership_enabled
                            and self.service_owner_side is None
                        ):
                            self._set_service_owner(
                                detected_server_side,
                                source="first_serve_detected",
                            )
                            frame_events.append("service_owner_initialized")
                        serve_latency_ms = self._record_decision_latency(
                            "serve_confirmed",
                            evidence_timestamp_s=first_sample.timestamp_s,
                            decision_timestamp_s=now_s,
                        )
                        self._start_rally(
                            origin,
                            first_sample.frame_index,
                            first_sample.timestamp_s,
                            decision_latency_ms=serve_latency_ms,
                            server_player_centers_court_xy=(
                                self._frame_eligible_player_centers_court_xy
                            ),
                            server_player_center_outside_court=(
                                self._frame_eligible_player_center_outside_court
                            ),
                        )
                        frame_events.append("serve_confirmed")
                        frame_evidence.extend(
                            [
                                "consecutive_observed_court_positions",
                                "netward_displacement_confirmed",
                            ]
                        )
                        if self.require_server_player_center_outside_court:
                            frame_evidence.append(
                                "eligible_server_player_center_outside_court"
                            )
                        if self.service_ownership_enabled:
                            frame_evidence.append(
                                "service_owner_side_match"
                            )
                        if self.expected_server_side is not None:
                            if detected_server_side == self.expected_server_side:
                                frame_evidence.append(
                                    "expected_server_side_match"
                                )
                            else:
                                frame_warnings.append(
                                    "expected_server_side_mismatch"
                                )
            elif self.phase == SERVE_CONFIRMING:
                if self._serve_samples and (
                    now_s - self._serve_samples[-1].timestamp_s > self.max_observation_gap_s
                ):
                    self._serve_samples.clear()
                    self.phase = WAIT_SERVE
                    frame_warnings.append("serve_confirmation_expired")

        elif self.phase == IN_RALLY:
            late_return_recovered = False
            if sample is not None:
                if self._net_crossed:
                    late_return_recovered = self._update_late_return_crossing(
                        sample,
                        allow_sample=discontinuity_reason is None,
                        frame_events=frame_events,
                        frame_evidence=frame_evidence,
                    )
                else:
                    self._update_crossing(sample, frame_events, frame_evidence)

            if "bounce_candidate" in event_names:
                fingerprint = self._event_fingerprint(event, event_names, current_frame)
                if self._remember_event(fingerprint):
                    rally_result = self._handle_bounce(
                        event,
                        event_names,
                        current_frame,
                        now_s,
                        frame_events,
                        frame_evidence,
                        frame_warnings,
                    )
                else:
                    self._duplicate_event_candidates += 1
                    frame_warnings.append("duplicate_bounce_candidate_ignored")

            if rally_result is None and self.phase == IN_RALLY:
                if (
                    sample is not None
                    and self._net_crossed
                    and not late_return_recovered
                ):
                    self._update_return_route(
                        sample,
                        now_s,
                        frame_events,
                        frame_evidence,
                    )
                if (
                    self._pending_bounce is not None
                    and self.pending_bounce_timeout_s > 0.0
                    and now_s - self._pending_bounce.timestamp_s
                    > self.pending_bounce_timeout_s
                ):
                    pending = self._pending_bounce
                    rally_result = self._finish_rally(
                        terminal_event="unknown",
                        fault_half=None,
                        winner_half=None,
                        status="unresolved",
                        confidence=0.0,
                        frame_index=current_frame,
                        timestamp_s=now_s,
                        evidence=tuple(
                            dict.fromkeys(
                                (
                                    *pending.evidence,
                                    "pending_bounce_timeout",
                                )
                            )
                        ),
                        warnings=tuple(
                            dict.fromkeys(
                                (
                                    *pending.quality_warnings,
                                    pending.reason,
                                    "unknown_does_not_change_score",
                                )
                            )
                        ),
                        evidence_timestamp_s=pending.timestamp_s,
                    )
                    frame_events.append("rally_ended_unknown")
                    frame_warnings.extend(
                        [pending.reason, "unknown_does_not_change_score"]
                    )
                if (
                    rally_result is None
                    and
                    self.unavailable_timeout_s > 0.0
                    and isfinite(self._last_observed_time_s)
                    and now_s - self._last_observed_time_s > self.unavailable_timeout_s
                ):
                    rally_result = self._finish_rally(
                        terminal_event="unknown",
                        fault_half=None,
                        winner_half=None,
                        status="unresolved",
                        confidence=0.0,
                        frame_index=current_frame,
                        timestamp_s=now_s,
                        evidence=("global_ball_unavailable_timeout",),
                        warnings=("unknown_does_not_change_score",),
                        evidence_timestamp_s=self._last_observed_time_s,
                    )
                    frame_events.append("rally_ended_unknown")
                    frame_warnings.append("unknown_does_not_change_score")

        return self._snapshot(
            current_frame,
            now_s,
            physical_half=physical_half,
            events=frame_events,
            evidence=frame_evidence,
            warnings=frame_warnings,
            rally_result=rally_result,
        )

    def diagnostics(self) -> dict:
        completed = self._demo_inferred_results + self._unknown_results
        return {
            "enabled": self.enabled,
            "phase": self.phase,
            "require_serve_arming": self.require_serve_arming,
            "serve_armed": self._serve_armed,
            "serve_arming_stable_count": self._serve_arming_stable_count,
            "serve_arming_missing_count": self._serve_arming_missing_count,
            "service_ownership_enabled": self.service_ownership_enabled,
            "service_owner_half": self._service_owner_half(),
            "service_owner_side": self.service_owner_side,
            "service_owner_source": self.service_owner_source,
            "service_owner_changes": self._service_owner_changes,
            "wrong_server_rejections": self._wrong_server_rejections,
            "require_server_player_center_outside_court": (
                self.require_server_player_center_outside_court
            ),
            "server_position_rejections": self._server_position_rejections,
            "eligible_player_centers_court_xy": [
                list(point)
                for point in self._frame_eligible_player_centers_court_xy
            ],
            "eligible_player_center_outside_court": (
                self._frame_eligible_player_center_outside_court
            ),
            "expected_server_side": self.expected_server_side,
            "detected_server_side": (
                self.half_to_side.get(self._server_half)
                if self._server_half is not None
                else None
            ),
            "server_side_match": self._server_side_match(),
            "current_rally_id": self._current_rally_id,
            "pending_bounce": self._pending_bounce is not None,
            "pending_bounce_reason": (
                self._pending_bounce.reason
                if self._pending_bounce is not None
                else None
            ),
            "updates": self._updates,
            "observed_samples": self._observed_samples,
            "serves_confirmed": self._serves_confirmed,
            "rallies_started": self._rally_sequence,
            "rallies_completed": completed,
            "unresolved_rallies": self._unknown_results,
            "routes_confirmed": self._routes_confirmed,
            "crossings_confirmed": self._crossings_confirmed,
            "demo_inferred_results": self._demo_inferred_results,
            "unknown_results": self._unknown_results,
            "duplicate_event_candidates": self._duplicate_event_candidates,
            "terminal_events": dict(sorted(self._terminal_events.items())),
            "decision_latency_ms": {
                name: {
                    "count": int(values["count"]),
                    "average": (
                        float(values["total"]) / int(values["count"])
                        if int(values["count"]) > 0
                        else 0.0
                    ),
                    "max": float(values["max"]),
                }
                for name, values in sorted(self._decision_latency_stats.items())
            },
            "score": self.score_recorder.score,
            "scoring": self.score_recorder.diagnostics(),
        }

    def _server_side_match(self) -> bool | None:
        if self.expected_server_side is None or self._server_half is None:
            return None
        return self.half_to_side[self._server_half] == self.expected_server_side

    def _record_decision_latency(
        self,
        name: str,
        *,
        evidence_timestamp_s: float,
        decision_timestamp_s: float,
    ) -> float:
        latency_ms = max(
            0.0,
            (float(decision_timestamp_s) - float(evidence_timestamp_s)) * 1000.0,
        )
        latency_ms = round(latency_ms, 3)
        self._frame_decision_latency_ms[str(name)] = latency_ms
        stats = self._decision_latency_stats.setdefault(
            str(name),
            {"count": 0, "total": 0.0, "max": 0.0},
        )
        stats["count"] = int(stats["count"]) + 1
        stats["total"] = float(stats["total"]) + latency_ms
        stats["max"] = max(float(stats["max"]), latency_ms)
        return latency_ms

    def _start_hit_record(
        self,
        *,
        origin_half: str,
        destination_half: str,
        frame_index: int,
        timestamp_s: float,
        decision_latency_ms: float | None,
    ) -> None:
        state = _HitRecordState(
            hit_count=len(self._hit_records) + 1,
            hit_half=origin_half,
            hit_side=self.half_to_side[origin_half],
            route_destination_half=destination_half,
            route_destination_side=self.half_to_side[destination_half],
            hit_start_frame_index=int(frame_index),
            hit_start_timestamp_s=float(timestamp_s),
            decision_latency_ms=decision_latency_ms,
        )
        self._hit_records.append(state)
        self._current_hit_record = state

    def _close_current_hit(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        event_type_0811: str,
    ) -> None:
        record = self._current_hit_record
        if record is None:
            return
        if record.hit_end_frame_index is None:
            record.hit_end_frame_index = int(frame_index)
            record.hit_end_timestamp_s = float(timestamp_s)
        if event_type_0811 != "NONE":
            record.event_type_0811 = str(event_type_0811)

    def _record_first_landing(
        self,
        *,
        frame_index: int,
        timestamp_s: float,
        contact_xy: tuple[float, float] | None,
        contact_inside: bool | None,
    ) -> None:
        record = self._current_hit_record
        if record is None or record.first_landing_frame_index is not None:
            return
        record.first_landing_frame_index = int(frame_index)
        record.first_landing_timestamp_s = float(timestamp_s)
        record.first_landing_position = contact_xy
        if contact_inside is True:
            record.first_landing_inout = "IN"
        elif contact_inside is False:
            record.first_landing_inout = "OUT"
        else:
            record.first_landing_inout = "UNKNOWN"

    def _hit_record_snapshots(self) -> list[HitRecord]:
        return [record.snapshot() for record in self._hit_records]

    def _observed_sample(
        self,
        projection: ProjectionResult,
        timestamp_s: float,
        frame_index: int,
    ) -> _ObservedSample | None:
        point = projection.ball_court_xy
        if not projection.projection_valid or not projection.observed or point is None or len(point) < 2:
            return None
        x_ft, y_ft = float(point[0]), float(point[1])
        if not isfinite(x_ft) or not isfinite(y_ft):
            return None
        margin = self.observation_margin_ft
        if not (
            -margin <= x_ft <= self.layout.width_ft + margin
            and -margin <= y_ft <= self.layout.length_ft + margin
        ):
            return None
        half = self._physical_half(y_ft)
        if half is None:
            return None
        return _ObservedSample(
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            court_xy=(x_ft, y_ft),
            half=half,
        )

    @staticmethod
    def _is_direct_observed_projection(projection: ProjectionResult) -> bool:
        point = projection.ball_court_xy
        if (
            not projection.projection_valid
            or not projection.observed
            or point is None
            or len(point) < 2
        ):
            return False
        try:
            return isfinite(float(point[0])) and isfinite(float(point[1]))
        except (TypeError, ValueError):
            return False

    def _update_serve_arming(
        self,
        sample: _ObservedSample | None,
        *,
        direct_observed: bool,
        predicted: bool,
        timestamp_s: float,
    ) -> str | None:
        """Arm initial serve detection only after stable or missing evidence."""
        if self._serve_armed:
            return None
        if predicted:
            self._serve_arming_stable_samples.clear()
            self._serve_arming_missing_since_s = None
            return None
        if not direct_observed:
            self._serve_arming_stable_samples.clear()
            if self._serve_arming_missing_since_s is None:
                self._serve_arming_missing_since_s = timestamp_s
            if (
                timestamp_s - self._serve_arming_missing_since_s
                >= self.post_rally_rearm_missing_s
            ):
                self._serve_armed = True
                self._serve_arming_missing_count += 1
                self._serve_samples.clear()
                return "initial_ball_absence_serve_armed"
            return None

        self._serve_arming_missing_since_s = None
        if sample is None:
            self._serve_arming_stable_samples.clear()
            return None
        self._append_contiguous(self._serve_arming_stable_samples, sample)
        if (
            len(self._serve_arming_stable_samples)
            >= self.post_rally_rearm_stable_frames
            and self._samples_are_stable_2d(self._serve_arming_stable_samples)
        ):
            self._serve_armed = True
            self._serve_arming_stable_count += 1
            self._serve_samples.clear()
            return "initial_stable_ball_serve_armed"
        return None

    def _samples_are_stable_2d(
        self,
        samples: deque[_ObservedSample],
    ) -> bool:
        if not samples:
            return False
        x_values = [item.court_xy[0] for item in samples]
        y_values = [item.court_xy[1] for item in samples]
        displacement = hypot(
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
        )
        return displacement <= self.post_rally_rearm_stable_displacement_ft

    def _update_guard_rearm(
        self,
        sample: _ObservedSample | None,
        *,
        direct_observed: bool,
        timestamp_s: float,
        frame_evidence: list[str],
    ) -> None:
        if self._guard_rearmed:
            return
        if not direct_observed:
            self._guard_stable_samples.clear()
            if (
                timestamp_s >= self._guard_until_s
                and
                isfinite(self._last_observed_time_s)
                and timestamp_s - self._last_observed_time_s
                >= self.post_rally_rearm_missing_s
            ):
                self._guard_rearmed = True
                frame_evidence.append("post_rally_ball_absence_rearmed")
            return
        if sample is None:
            self._guard_stable_samples.clear()
            return
        self._append_contiguous(self._guard_stable_samples, sample)
        if len(self._guard_stable_samples) < self.post_rally_rearm_stable_frames:
            return
        if (
            timestamp_s >= self._guard_until_s
            and self._samples_are_stable_2d(self._guard_stable_samples)
        ):
            self._guard_rearmed = True
            frame_evidence.append("post_rally_stable_ball_rearmed")

    def _physical_half(self, y_ft: float) -> str | None:
        lower = self.layout.net_y_ft - self.net_deadband_ft
        upper = self.layout.net_y_ft + self.net_deadband_ft
        if y_ft < lower:
            return LOW_Y_HALF
        if y_ft > upper:
            return HIGH_Y_HALF
        return None

    def _append_contiguous(
        self,
        samples: deque[_ObservedSample],
        sample: _ObservedSample,
        *,
        window_s: float | None = None,
    ) -> None:
        if samples and (
            sample.half != samples[-1].half
            or sample.timestamp_s <= samples[-1].timestamp_s
            or sample.timestamp_s - samples[-1].timestamp_s > self.max_observation_gap_s
        ):
            samples.clear()
        samples.append(sample)
        if window_s is not None:
            cutoff_s = sample.timestamp_s - max(0.0, float(window_s))
            time_epsilon_s = 1e-9
            while (
                len(samples) > 1
                and samples[0].timestamp_s < cutoff_s - time_epsilon_s
            ):
                samples.popleft()

    @staticmethod
    def _opposite_half(half: str) -> str:
        if half == LOW_Y_HALF:
            return HIGH_Y_HALF
        if half == HIGH_Y_HALF:
            return LOW_Y_HALF
        raise ValueError(f"unknown physical half: {half!r}")

    def _confirms_netward_route(
        self,
        samples: deque[_ObservedSample],
        origin_half: str,
        required_frames: int,
        min_displacement_ft: float,
    ) -> bool:
        return bool(
            self._directional_confirmation_window(
                samples,
                origin_half,
                toward_net=True,
                required_frames=required_frames,
                min_displacement_ft=min_displacement_ft,
            )
        )

    def _directional_confirmation_window(
        self,
        samples: deque[_ObservedSample],
        half: str,
        *,
        toward_net: bool,
        required_frames: int,
        min_displacement_ft: float,
    ) -> tuple[_ObservedSample, ...]:
        """Return a recent monotonic time-window suffix that confirms motion."""
        if len(samples) < required_frames:
            return ()
        items = list(samples)
        if items[-1].half != half:
            return ()
        direction_epsilon = 0.05
        start_index = len(items) - 1
        for index in range(len(items) - 1, 0, -1):
            before = items[index - 1]
            after = items[index]
            if before.half != half or after.half != half:
                break
            toward_high_y = (half == LOW_Y_HALF) if toward_net else (half == HIGH_Y_HALF)
            step = (
                after.court_xy[1] - before.court_xy[1]
                if toward_high_y
                else before.court_xy[1] - after.court_xy[1]
            )
            if step < -direction_epsilon:
                break
            start_index = index - 1
        window = tuple(items[start_index:])
        if len(window) < required_frames:
            return ()
        if (
            self.motion_confirmation_window_s > 0.0
            and window[-1].timestamp_s - window[0].timestamp_s
            > self.motion_confirmation_window_s + 1e-9
        ):
            return ()
        y_start = window[0].court_xy[1]
        y_end = window[-1].court_xy[1]
        toward_high_y = (half == LOW_Y_HALF) if toward_net else (half == HIGH_Y_HALF)
        displacement = y_end - y_start if toward_high_y else y_start - y_end
        if displacement < min_displacement_ft:
            return ()
        return window

    def _start_rally(
        self,
        origin_half: str,
        frame_index: int,
        timestamp_s: float,
        *,
        decision_latency_ms: float | None = None,
        server_player_centers_court_xy=(),
        server_player_center_outside_court: bool | None = None,
        route_origin_half: str | None = None,
        leg_index: int = 1,
        confirmed_serve: bool = True,
    ) -> None:
        active_route_origin = route_origin_half or origin_half
        self._rally_sequence += 1
        self._current_rally_id = self._rally_sequence
        self._rally_start_frame = frame_index
        self._rally_start_time_s = timestamp_s
        self._server_half = origin_half
        self._server_player_centers_court_xy = tuple(
            self._finite_player_centers(server_player_centers_court_xy)
        )
        self._server_player_center_outside_court = (
            bool(server_player_center_outside_court)
            if server_player_center_outside_court is not None
            else None
        )
        self._serve_armed = not self.require_serve_arming
        self._serve_arming_missing_since_s = None
        self._serve_arming_stable_samples.clear()
        self._route_origin_half = active_route_origin
        self._route_destination_half = self._opposite_half(active_route_origin)
        self._leg_index = max(1, int(leg_index))
        self._net_crossed = False
        self._first_bounce_half = None
        self._hit_records = []
        self._current_hit_record = None
        self._start_hit_record(
            origin_half=active_route_origin,
            destination_half=self._opposite_half(active_route_origin),
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            decision_latency_ms=decision_latency_ms,
        )
        self._pending_bounce = None
        self._cross_samples.clear()
        self._return_samples.clear()
        self._late_cross_samples.clear()
        self._serve_samples.clear()
        self.phase = IN_RALLY
        if confirmed_serve:
            self._serves_confirmed += 1
        self._routes_confirmed += 1

    def _update_crossing(
        self,
        sample: _ObservedSample,
        frame_events: list[str],
        frame_evidence: list[str],
    ) -> None:
        if self._net_crossed or self._route_destination_half is None:
            return
        if sample.half != self._route_destination_half:
            self._cross_samples.clear()
            return
        self._append_contiguous(
            self._cross_samples,
            sample,
            window_s=self.motion_confirmation_window_s,
        )
        if self._confirms_away_from_net(
            self._cross_samples,
            self._route_destination_half,
            self.cross_confirm_frames,
            self.cross_min_displacement_ft,
        ):
            self._net_crossed = True
            self._return_samples.clear()
            self._crossings_confirmed += 1
            self._record_decision_latency(
                "net_crossing_confirmed",
                evidence_timestamp_s=self._cross_samples[0].timestamp_s,
                decision_timestamp_s=sample.timestamp_s,
            )
            frame_events.append("net_crossing_confirmed")
            frame_evidence.append("consecutive_observed_opposite_half")

    def _confirms_away_from_net(
        self,
        samples: deque[_ObservedSample],
        destination_half: str,
        required_frames: int,
        min_displacement_ft: float,
    ) -> bool:
        return bool(
            self._directional_confirmation_window(
                samples,
                destination_half,
                toward_net=False,
                required_frames=required_frames,
                min_displacement_ft=min_displacement_ft,
            )
        )

    def _update_late_return_crossing(
        self,
        sample: _ObservedSample,
        *,
        allow_sample: bool,
        frame_events: list[str],
        frame_evidence: list[str],
    ) -> bool:
        """Recover a missed return only after a fresh safe opposite-side trace.

        A discontinuity frame never contributes.  After any reset, the first
        accepted sample must enter the old origin close to the deadband; this
        prevents a deep-half track jump from being interpreted as a return.
        """
        origin = self._route_origin_half
        destination = self._route_destination_half
        if not allow_sample or origin is None or destination is None:
            self._late_cross_samples.clear()
            return False
        if sample.half != origin:
            self._late_cross_samples.clear()
            return False
        if not self._late_cross_samples:
            safe_boundary_y = (
                self.layout.net_y_ft + self.net_deadband_ft
                if origin == HIGH_Y_HALF
                else self.layout.net_y_ft - self.net_deadband_ft
            )
            entry_distance_ft = abs(sample.court_xy[1] - safe_boundary_y)
            max_entry_distance_ft = max(
                self.route_min_displacement_ft,
                self.cross_min_displacement_ft,
            )
            if entry_distance_ft > max_entry_distance_ft:
                return False
        self._append_contiguous(
            self._late_cross_samples,
            sample,
            window_s=self.motion_confirmation_window_s,
        )
        if not self._confirms_away_from_net(
            self._late_cross_samples,
            origin,
            self.cross_confirm_frames,
            self.cross_min_displacement_ft,
        ):
            return False

        previous_origin = origin
        previous_destination = destination
        previous_had_bounce = self._first_bounce_half is not None
        previous_was_volleyed = (
            not previous_had_bounce and self._pending_bounce is None
        )
        self._close_current_hit(
            frame_index=sample.frame_index,
            timestamp_s=sample.timestamp_s,
            event_type_0811=(
                "firstlanding_in"
                if previous_had_bounce
                else ("volleyed" if previous_was_volleyed else "NONE")
            ),
        )
        self._route_origin_half = previous_destination
        self._route_destination_half = previous_origin
        self._leg_index += 1
        self._net_crossed = True
        self._first_bounce_half = None
        route_latency_ms = self._record_decision_latency(
            "return_route_confirmed_late",
            evidence_timestamp_s=sample.timestamp_s,
            decision_timestamp_s=sample.timestamp_s,
        )
        self._record_decision_latency(
            "net_crossing_confirmed_late",
            evidence_timestamp_s=sample.timestamp_s,
            decision_timestamp_s=sample.timestamp_s,
        )
        self._start_hit_record(
            origin_half=previous_destination,
            destination_half=previous_origin,
            frame_index=sample.frame_index,
            timestamp_s=sample.timestamp_s,
            decision_latency_ms=route_latency_ms,
        )
        if (
            self._pending_bounce is not None
            and self._pending_bounce.contact_half == self._route_destination_half
            and self._pending_bounce.contact_inside is not False
        ):
            self._first_bounce_half = self._route_destination_half
            frame_evidence.append("pending_bounce_adopted_after_late_route_recovery")
        self._pending_bounce = None
        self._cross_samples.clear()
        self._return_samples.clear()
        self._late_cross_samples.clear()
        self._routes_confirmed += 1
        self._crossings_confirmed += 1
        frame_events.extend(
            ["return_route_confirmed_late", "net_crossing_confirmed_late"]
        )
        if previous_was_volleyed:
            frame_events.append("volleyed_return")
        frame_evidence.extend(
            [
                "fresh_consecutive_observed_old_origin_samples",
                "safe_deadband_exit_confirmed",
                "late_recovery_sequence_has_no_discontinuity",
            ]
        )
        return True

    def _update_return_route(
        self,
        sample: _ObservedSample,
        timestamp_s: float,
        frame_events: list[str],
        frame_evidence: list[str],
    ) -> None:
        destination = self._route_destination_half
        if destination is None or sample.half != destination:
            self._return_samples.clear()
            return
        self._append_contiguous(
            self._return_samples,
            sample,
            window_s=self.motion_confirmation_window_s,
        )
        confirmation_window = self._directional_confirmation_window(
            self._return_samples,
            destination,
            toward_net=True,
            required_frames=self.route_confirm_frames,
            min_displacement_ft=self.route_min_displacement_ft,
        )
        if not confirmation_window:
            return
        next_hit_start = confirmation_window[0]
        previous_origin = self._route_origin_half
        previous_had_bounce = self._first_bounce_half is not None
        previous_was_volleyed = (
            not previous_had_bounce and self._pending_bounce is None
        )
        self._close_current_hit(
            frame_index=next_hit_start.frame_index,
            timestamp_s=next_hit_start.timestamp_s,
            event_type_0811=(
                "firstlanding_in"
                if previous_had_bounce
                else ("volleyed" if previous_was_volleyed else "NONE")
            ),
        )
        self._route_origin_half = destination
        self._route_destination_half = previous_origin
        self._leg_index += 1
        self._net_crossed = False
        self._first_bounce_half = None
        route_latency_ms = self._record_decision_latency(
            "return_route_confirmed",
            evidence_timestamp_s=next_hit_start.timestamp_s,
            decision_timestamp_s=timestamp_s,
        )
        self._start_hit_record(
            origin_half=destination,
            destination_half=previous_origin,
            frame_index=next_hit_start.frame_index,
            timestamp_s=next_hit_start.timestamp_s,
            decision_latency_ms=route_latency_ms,
        )
        if self._pending_bounce is not None:
            self._pending_bounce = None
            frame_evidence.append(
                "pending_bounce_superseded_by_confirmed_return_route"
            )
        self._cross_samples.clear()
        self._return_samples.clear()
        self._late_cross_samples.clear()
        self._routes_confirmed += 1
        frame_events.append("return_route_confirmed")
        if previous_was_volleyed:
            frame_events.append("volleyed_return")
        frame_evidence.extend(
            [
                "consecutive_observed_court_positions",
                "netward_direction_reversal_confirmed",
            ]
        )
        if (
            isfinite(self._last_hit_candidate_time_s)
            and timestamp_s - self._last_hit_candidate_time_s <= self.hit_assist_window_s
        ):
            frame_evidence.append("recent_paddle_hit_candidate_support")

    def _handle_bounce(
        self,
        event: dict,
        event_names: set[str],
        frame_index: int,
        timestamp_s: float,
        frame_events: list[str],
        frame_evidence: list[str],
        frame_warnings: list[str],
    ) -> RallyResult | None:
        contact_xy = event.get("contact_court_xy")
        contact_frame_index = int(event.get("contact_frame_index", frame_index))
        try:
            contact_timestamp_s = float(
                event.get("contact_timestamp_s", timestamp_s)
            )
        except (TypeError, ValueError):
            contact_timestamp_s = float(timestamp_s)
        if not isfinite(contact_timestamp_s):
            contact_timestamp_s = float(timestamp_s)
        normalized_contact_xy: tuple[float, float] | None = None
        contact_half: str | None = None
        if contact_xy is not None and len(contact_xy) >= 2:
            x_ft = float(contact_xy[0])
            y_ft = float(contact_xy[1])
            if isfinite(x_ft) and isfinite(y_ft):
                normalized_contact_xy = (x_ft, y_ft)
                contact_half = self._physical_half(y_ft)

        origin = self._route_origin_half
        destination = self._route_destination_half
        if origin is None or destination is None:
            return None

        base_evidence = [
            "bounce_candidate",
        ]
        if normalized_contact_xy is not None:
            base_evidence.append("contact_court_xy_used")
        base_evidence.extend(str(item) for item in event.get("evidence", []))
        quality_warnings = tuple(
            dict.fromkeys(str(item) for item in event.get("warnings", []))
        )
        frame_warnings.extend(
            warning for warning in quality_warnings if warning not in frame_warnings
        )
        contact_inside = event.get("contact_inside_court")

        if self._pending_bounce is not None:
            handled, result = self._handle_pending_bounce_followup(
                contact_half=contact_half,
                contact_inside=contact_inside,
                event_names=event_names,
                base_evidence=base_evidence,
                quality_warnings=quality_warnings,
                contact_timestamp_s=contact_timestamp_s,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                frame_events=frame_events,
                frame_evidence=frame_evidence,
                frame_warnings=frame_warnings,
            )
            if handled:
                return result
            self._pending_bounce = None
            frame_evidence.append("pending_bounce_superseded_by_later_contact")

        if (
            contact_half is None
            and normalized_contact_xy is not None
            and not self._net_crossed
        ):
            # A grounded contact inside the configured net deadband cannot be
            # assigned to either half.  When the current route has not crossed
            # the net, settle it as the route origin's net fault instead of
            # deferring an obvious under-net landing to an unknown timeout.
            terminal_event = "serve_net" if self._leg_index == 1 else "return_net"
            frame_events.append(terminal_event)
            self._record_first_landing(
                frame_index=contact_frame_index,
                timestamp_s=contact_timestamp_s,
                contact_xy=normalized_contact_xy,
                contact_inside=contact_inside,
            )
            return self._finish_rally(
                terminal_event=terminal_event,
                fault_half=origin,
                winner_half=destination,
                status="demo_inferred",
                confidence=0.8,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence=tuple(
                    base_evidence
                    + ["net_deadband_landing_without_confirmed_crossing"]
                ),
                warnings=tuple(
                    dict.fromkeys(
                        (*quality_warnings, "net_contact_not_directly_detected")
                    )
                ),
                evidence_timestamp_s=contact_timestamp_s,
                event_type_0811="off_net",
            )

        if contact_half is None:
            self._remember_pending_bounce(
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                contact_half=None,
                contact_xy=normalized_contact_xy,
                contact_inside=contact_inside,
                reason="bounce_in_net_deadband_or_missing_contact_coordinate",
                evidence=tuple(base_evidence),
                quality_warnings=quality_warnings,
            )
            frame_events.append("bounce_pending_uncertain")
            frame_warnings.extend(
                [
                    "bounce_contact_half_unknown",
                    "bounce_in_net_deadband_or_missing_contact_coordinate",
                ]
            )
            return None

        # Priority 1: a route that lands on its origin side without a confirmed
        # crossing is a serve/return net fault, regardless of an OUT flag.
        if not self._net_crossed and contact_half == origin:
            terminal_event = "serve_net" if self._leg_index == 1 else "return_net"
            frame_events.append(terminal_event)
            self._record_first_landing(
                frame_index=contact_frame_index,
                timestamp_s=contact_timestamp_s,
                contact_xy=normalized_contact_xy,
                contact_inside=contact_inside,
            )
            return self._finish_rally(
                terminal_event=terminal_event,
                fault_half=origin,
                winner_half=destination,
                status="demo_inferred",
                confidence=0.8,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence=tuple(base_evidence + ["origin_half_landing_without_confirmed_crossing"]),
                warnings=tuple(
                    dict.fromkeys(
                        (*quality_warnings, "net_contact_not_directly_detected")
                    )
                ),
                evidence_timestamp_s=contact_timestamp_s,
                event_type_0811="off_net",
            )

        # A bounce beyond the net before enough observed samples confirmed the
        # crossing remains pending so later evidence can settle it.
        if not self._net_crossed:
            reason = "opposite_half_bounce_without_confirmed_crossing"
            self._remember_pending_bounce(
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                contact_half=contact_half,
                contact_xy=normalized_contact_xy,
                contact_inside=contact_inside,
                reason=reason,
                evidence=tuple(base_evidence),
                quality_warnings=quality_warnings,
            )
            frame_events.append("bounce_pending_uncertain")
            frame_warnings.append(reason)
            return None

        if contact_half != destination:
            reason = "bounce_half_conflicts_with_current_route"
            self._remember_pending_bounce(
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                contact_half=contact_half,
                contact_xy=normalized_contact_xy,
                contact_inside=contact_inside,
                reason=reason,
                evidence=tuple(base_evidence),
                quality_warnings=quality_warnings,
            )
            frame_events.append("bounce_pending_uncertain")
            frame_warnings.append(reason)
            return None

        # Keep the existing event rule: an OUT contact takes precedence over a
        # simultaneous second-bounce candidate.
        if contact_inside is False or "out_of_bounds_bounce_candidate" in event_names:
            frame_events.append("out_of_bounds")
            self._record_first_landing(
                frame_index=contact_frame_index,
                timestamp_s=contact_timestamp_s,
                contact_xy=normalized_contact_xy,
                contact_inside=False,
            )
            return self._finish_rally(
                terminal_event="out_of_bounds",
                fault_half=origin,
                winner_half=destination,
                status="demo_inferred",
                confidence=0.75,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence=tuple(base_evidence + ["first_landing_outside_court"]),
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *quality_warnings,
                            "ground_plane_projection_requires_validation",
                        )
                    )
                ),
                evidence_timestamp_s=contact_timestamp_s,
                event_type_0811="firstlanding_out",
            )

        if self._first_bounce_half is not None:
            frame_events.append("second_bounce")
            evidence = base_evidence + ["prior_first_bounce_same_route"]
            if "second_bounce_candidate" in event_names:
                evidence.append("court_event_second_bounce_support")
            return self._finish_rally(
                terminal_event="second_bounce",
                fault_half=destination,
                winner_half=origin,
                status="demo_inferred",
                confidence=0.8,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence=tuple(evidence),
                warnings=quality_warnings,
                evidence_timestamp_s=contact_timestamp_s,
                event_type_0811="doublelanding_in",
            )

        self._first_bounce_half = contact_half
        self._record_first_landing(
            frame_index=contact_frame_index,
            timestamp_s=contact_timestamp_s,
            contact_xy=normalized_contact_xy,
            contact_inside=contact_inside,
        )
        self._record_decision_latency(
            "first_bounce_recorded",
            evidence_timestamp_s=contact_timestamp_s,
            decision_timestamp_s=timestamp_s,
        )
        frame_events.append("first_bounce_recorded")
        frame_evidence.extend(base_evidence)
        if contact_inside is not True:
            frame_events.append("first_bounce_inside_unknown")
            frame_warnings.append("first_bounce_inside_state_unknown")
        return None

    def _remember_pending_bounce(
        self,
        *,
        timestamp_s: float,
        frame_index: int,
        contact_half: str | None,
        contact_xy: tuple[float, float] | None,
        contact_inside: bool | None,
        reason: str,
        evidence: tuple[str, ...],
        quality_warnings: tuple[str, ...],
    ) -> None:
        self._pending_bounce = _PendingBounce(
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            contact_half=contact_half,
            contact_xy=contact_xy,
            contact_inside=contact_inside,
            reason=reason,
            evidence=evidence,
            quality_warnings=quality_warnings,
        )

    def _handle_pending_bounce_followup(
        self,
        *,
        contact_half: str | None,
        contact_inside: bool | None,
        event_names: set[str],
        base_evidence: list[str],
        quality_warnings: tuple[str, ...],
        contact_timestamp_s: float,
        frame_index: int,
        timestamp_s: float,
        frame_events: list[str],
        frame_evidence: list[str],
        frame_warnings: list[str],
    ) -> tuple[bool, RallyResult | None]:
        pending = self._pending_bounce
        if pending is None:
            return False, None
        if contact_half is None:
            frame_events.append("bounce_pending_uncertain")
            frame_warnings.append("additional_bounce_contact_half_unknown")
            return True, None
        if pending.contact_half != contact_half:
            return False, None

        origin = self._route_origin_half
        destination = self._route_destination_half
        if origin is None or destination is None:
            return True, None

        inference_evidence: list[str] = []
        if not self._net_crossed and contact_half == destination:
            self._net_crossed = True
            self._crossings_confirmed += 1
            inference_evidence.append(
                "crossing_inferred_from_repeated_destination_bounces"
            )
        elif self._net_crossed and contact_half != destination:
            previous_origin = origin
            previous_destination = destination
            previous_had_bounce = self._first_bounce_half is not None
            self._close_current_hit(
                frame_index=pending.frame_index,
                timestamp_s=pending.timestamp_s,
                event_type_0811=(
                    "firstlanding_in" if previous_had_bounce else "volleyed"
                ),
            )
            self._route_origin_half = previous_destination
            self._route_destination_half = previous_origin
            self._leg_index += 1
            origin = self._route_origin_half
            destination = self._route_destination_half
            inferred_latency_ms = self._record_decision_latency(
                "return_route_confirmed_inferred",
                evidence_timestamp_s=pending.timestamp_s,
                decision_timestamp_s=timestamp_s,
            )
            self._start_hit_record(
                origin_half=origin,
                destination_half=destination,
                frame_index=pending.frame_index,
                timestamp_s=pending.timestamp_s,
                decision_latency_ms=inferred_latency_ms,
            )
            if not previous_had_bounce:
                frame_events.append("volleyed_return")
            self._routes_confirmed += 1
            self._crossings_confirmed += 1
            inference_evidence.append(
                "missed_return_inferred_from_repeated_conflict_bounces"
            )

        if contact_half != destination:
            frame_events.append("bounce_pending_uncertain")
            frame_warnings.append("pending_bounce_not_yet_resolvable")
            return True, None

        combined_evidence = list(
            dict.fromkeys(
                (
                    *pending.evidence,
                    *base_evidence,
                    *inference_evidence,
                    "prior_pending_bounce_same_physical_half",
                )
            )
        )
        combined_warnings = tuple(
            dict.fromkeys(
                (
                    *pending.quality_warnings,
                    *quality_warnings,
                    pending.reason,
                )
            )
        )
        frame_evidence.extend(inference_evidence)
        frame_warnings.extend(
            warning for warning in combined_warnings if warning not in frame_warnings
        )
        self._pending_bounce = None
        self._record_first_landing(
            frame_index=pending.frame_index,
            timestamp_s=pending.timestamp_s,
            contact_xy=pending.contact_xy,
            contact_inside=pending.contact_inside,
        )

        # Keep the established OUT-over-second priority even when the first
        # contact was temporarily ambiguous.
        if (
            pending.contact_inside is False
            or contact_inside is False
            or "out_of_bounds_bounce_candidate" in event_names
        ):
            frame_events.append("out_of_bounds")
            return True, self._finish_rally(
                terminal_event="out_of_bounds",
                fault_half=origin,
                winner_half=destination,
                status="demo_inferred",
                confidence=0.75,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                evidence=tuple(
                    combined_evidence + ["first_landing_outside_court"]
                ),
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *combined_warnings,
                            "ground_plane_projection_requires_validation",
                        )
                    )
                ),
                evidence_timestamp_s=(
                    pending.timestamp_s
                    if pending.contact_inside is False
                    else contact_timestamp_s
                ),
                event_type_0811="firstlanding_out",
            )

        frame_events.append("second_bounce")
        if "second_bounce_candidate" in event_names:
            combined_evidence.append("court_event_second_bounce_support")
        return True, self._finish_rally(
            terminal_event="second_bounce",
            fault_half=destination,
            winner_half=origin,
            status="demo_inferred",
            confidence=0.8,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            evidence=tuple(combined_evidence),
            warnings=combined_warnings,
            evidence_timestamp_s=contact_timestamp_s,
            event_type_0811="doublelanding_in",
        )

    def _finish_rally(
        self,
        *,
        terminal_event: str,
        fault_half: str | None,
        winner_half: str | None,
        status: str,
        confidence: float,
        frame_index: int,
        timestamp_s: float,
        evidence: tuple[str, ...],
        warnings: tuple[str, ...],
        evidence_timestamp_s: float | None = None,
        event_type_0811: str | None = None,
    ) -> RallyResult:
        if self._current_rally_id is None or self._server_half is None:
            raise RuntimeError("cannot finish a rally that was not started")
        winner_side = self.half_to_side[winner_half] if winner_half in VALID_HALVES else None
        fault_side = self.half_to_side[fault_half] if fault_half in VALID_HALVES else None
        server_side = self.half_to_side[self._server_half]
        decision = self.score_recorder.award(
            self._current_rally_id,
            rally_winner=winner_side,
            server_side=server_side,
            status=status,
        )
        if (
            self.service_ownership_enabled
            and status in {"confirmed", "demo_inferred"}
            and winner_side in self.score_recorder.teams
        ):
            self._set_service_owner(
                winner_side,
                source="rally_winner",
            )
        next_server_side = (
            self.service_owner_side
            if self.service_ownership_enabled
            else None
        )
        next_server_half = (
            self._service_owner_half()
            if self.service_ownership_enabled
            else None
        )
        mapped_event_type = (
            map_event_type_0811((), terminal_event=terminal_event)
            if event_type_0811 is None
            else str(event_type_0811)
        )
        evidence_time_s = (
            float(timestamp_s)
            if evidence_timestamp_s is None
            else float(evidence_timestamp_s)
        )
        decision_latency_ms = self._record_decision_latency(
            terminal_event,
            evidence_timestamp_s=evidence_time_s,
            decision_timestamp_s=timestamp_s,
        )
        self._close_current_hit(
            frame_index=frame_index,
            timestamp_s=evidence_time_s,
            event_type_0811=mapped_event_type,
        )
        result = RallyResult(
            rally_id=self._current_rally_id,
            start_frame_index=self._rally_start_frame,
            start_timestamp_s=self._rally_start_time_s,
            end_frame_index=frame_index,
            end_timestamp_s=timestamp_s,
            server_half=self._server_half,
            server_side=server_side,
            leg_index=self._leg_index,
            terminal_event=terminal_event,
            fault_half=fault_half,
            fault_side=fault_side,
            rally_winner_half=winner_half,
            rally_winner=winner_side,
            point_awarded_to=decision.point_awarded_to,
            score_before=decision.score_before,
            score_after=decision.score_after,
            status=status,
            confidence=min(1.0, max(0.0, float(confidence))),
            evidence=evidence,
            warnings=tuple(
                dict.fromkeys(
                    (
                        *warnings,
                        "demo_heuristic_not_official_verdict",
                        "confidence_not_calibrated",
                    )
                )
            ),
            server_player_centers_court_xy=(
                self._server_player_centers_court_xy
            ),
            server_player_center_outside_court=(
                self._server_player_center_outside_court
            ),
            next_server_half=next_server_half,
            next_server_side=next_server_side,
            expected_server_side=self.expected_server_side,
            server_side_match=self._server_side_match(),
            event_type_0811=mapped_event_type,
            rally_over=rally_over_0811(
                phase=self.phase,
                terminal_status=status,
            ),
            manual_confirmation_required=status == "unresolved",
            decision_latency_ms=decision_latency_ms,
            hit_records=tuple(self._hit_record_snapshots()),
        )
        if status == "demo_inferred":
            self._demo_inferred_results += 1
        else:
            self._unknown_results += 1
        self._terminal_events[terminal_event] += 1
        self.phase = POST_RALLY_GUARD
        self._pending_bounce = None
        self._guard_until_s = timestamp_s + self.post_rally_guard_s
        self._guard_rearmed = False
        self._guard_stable_samples.clear()
        self._clear_motion_buffers()
        return result

    def _snapshot(
        self,
        frame_index: int,
        timestamp_s: float,
        *,
        physical_half: str | None,
        events: list[str],
        evidence: list[str],
        warnings: list[str],
        rally_result: RallyResult | None,
    ) -> RefereeFrameResult:
        def side(half: str | None) -> str | None:
            return self.half_to_side.get(half) if half is not None else None

        terminal_status = rally_result.status if rally_result is not None else None
        mapped_event_type = map_event_type_0811(
            events,
            terminal_event=(
                rally_result.terminal_event if rally_result is not None else None
            ),
        )
        return RefereeFrameResult(
            phase=self.phase,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            rally_id=self._current_rally_id,
            physical_half=physical_half,
            physical_side=side(physical_half),
            server_half=self._server_half,
            server_side=side(self._server_half),
            route_origin_half=self._route_origin_half,
            route_origin_side=side(self._route_origin_half),
            route_destination_half=self._route_destination_half,
            route_destination_side=side(self._route_destination_half),
            leg_index=self._leg_index,
            net_crossed=self._net_crossed,
            first_bounce_half=self._first_bounce_half,
            score=self.score_recorder.score,
            eligible_player_centers_court_xy=list(
                self._frame_eligible_player_centers_court_xy
            ),
            eligible_player_center_outside_court=(
                self._frame_eligible_player_center_outside_court
            ),
            service_owner_half=self._service_owner_half(),
            service_owner_side=self.service_owner_side,
            expected_server_side=self.expected_server_side,
            server_side_match=self._server_side_match(),
            target_ball_state_0811=self._frame_target_ball_state_0811,
            event_type_0811=mapped_event_type,
            rally_over=rally_over_0811(
                phase=self.phase,
                terminal_status=terminal_status,
            ),
            manual_confirmation_required=terminal_status == "unresolved",
            decision_latency_ms=dict(self._frame_decision_latency_ms),
            hit_records=self._hit_record_snapshots(),
            events=events,
            evidence=evidence,
            warnings=warnings,
            rally_result=rally_result,
        )

    @staticmethod
    def _event_fingerprint(
        event: dict,
        event_names: set[str],
        fallback_frame_index: int,
    ) -> tuple[Any, ...]:
        contact_xy = event.get("contact_court_xy")
        rounded_xy = None
        if contact_xy is not None and len(contact_xy) >= 2:
            rounded_xy = (
                round(float(contact_xy[0]), 3),
                round(float(contact_xy[1]), 3),
            )
        return (
            int(event.get("contact_frame_index", fallback_frame_index)),
            rounded_xy,
            tuple(sorted(event_names)),
        )

    def _remember_event(self, fingerprint: tuple[Any, ...]) -> bool:
        if fingerprint in self._event_fingerprint_set:
            return False
        if len(self._event_fingerprints) == self._event_fingerprints.maxlen:
            oldest = self._event_fingerprints.popleft()
            self._event_fingerprint_set.discard(oldest)
        self._event_fingerprints.append(fingerprint)
        self._event_fingerprint_set.add(fingerprint)
        return True

    def _clear_motion_buffers(self) -> None:
        self._serve_samples.clear()
        self._cross_samples.clear()
        self._return_samples.clear()
        self._late_cross_samples.clear()
        self._guard_stable_samples.clear()
        self._serve_arming_stable_samples.clear()
        self._waiting_return_hit_sample = None
        self._waiting_return_samples.clear()

    def _clear_current_rally(self) -> None:
        self._current_rally_id = None
        self._rally_start_frame = -1
        self._rally_start_time_s = 0.0
        self._server_half = None
        self._server_player_centers_court_xy = ()
        self._server_player_center_outside_court = None
        self._route_origin_half = None
        self._route_destination_half = None
        self._leg_index = 0
        self._net_crossed = False
        self._first_bounce_half = None
        self._hit_records = []
        self._current_hit_record = None
        self._pending_bounce = None
        self._waiting_return_hit_sample = None
        self._waiting_return_samples.clear()
