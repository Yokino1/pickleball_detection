"""Read-only referee-event candidates derived from accepted R9 outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import acos, degrees, hypot, isfinite
from typing import Any

from .projector import ProjectionResult


@dataclass
class CourtEventResult:
    """Per-frame event candidate and latched presentation state."""

    phase: str = "airborne"
    display_color: str = "fluorescent_green"
    display_state: str = "airborne"
    display_text_zh: str = "飞行"
    events: list[str] = field(default_factory=list)
    bounce_index_since_last_hit: int = 0
    candidate: bool = False
    fault_candidate: bool = False
    fault_reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    contact_frame_index: int | None = None
    contact_timestamp_s: float | None = None
    contact_image_xy: list[float] | None = None
    contact_court_xy: list[float] | None = None
    contact_inside_court: bool | None = None
    metrics: dict[str, float | bool | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "display_color": self.display_color,
            "display_state": self.display_state,
            "display_text_zh": self.display_text_zh,
            "events": list(self.events),
            "bounce_index_since_last_hit": self.bounce_index_since_last_hit,
            "candidate": self.candidate,
            "fault_candidate": self.fault_candidate,
            "fault_reasons": list(self.fault_reasons),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "contact_frame_index": self.contact_frame_index,
            "contact_timestamp_s": self.contact_timestamp_s,
            "contact_image_xy": (
                list(self.contact_image_xy)
                if self.contact_image_xy is not None
                else None
            ),
            "contact_court_xy": (
                list(self.contact_court_xy)
                if self.contact_court_xy is not None
                else None
            ),
            "contact_inside_court": self.contact_inside_court,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class _MotionSample:
    """One accepted global observation used only by the event interpreter."""

    timestamp_s: float
    frame_index: int
    image_xy: tuple[float, float]
    court_xy: tuple[float, float] | None
    inside_court: bool | None
    player_relation: str


@dataclass(frozen=True)
class _KinematicCandidate:
    """Multi-frame change measured around one observed turning point."""

    kind: str
    contact: _MotionSample
    pre_velocity: tuple[float, float]
    post_velocity: tuple[float, float]
    direction_change_deg: float
    impulse_speed_px_per_second: float
    horizontal_impulse_ratio: float


@dataclass(frozen=True)
class _DiscontinuityHitCandidate:
    """Hit evidence that remains usable across a local/global track boundary."""

    source: str
    contact: _MotionSample
    pre_velocity: tuple[float, float] | None
    post_velocity: tuple[float, float]
    direction_change_deg: float | None
    impulse_speed_px_per_second: float


class CourtEventInterpreter:
    """Interpret accepted R9 output without feeding anything back into R9.

    R9 bounce/impact recovery counters are association diagnostics rather than
    referee labels. This layer therefore combines them with a short history of
    accepted global observations and already-available eligible-player boxes.
    It never runs an additional detector and never changes a track.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        hit_flash_frames: int = 3,
        ground_min_hold_frames: int = 3,
        rise_confirm_frames: int = 2,
        rise_min_speed_px_per_second: float = 40.0,
        kinematic_enabled: bool = True,
        max_observation_gap_ms: float = 80.0,
        bounce_min_downward_speed_px_per_second: float = 100.0,
        bounce_min_upward_speed_px_per_second: float = 40.0,
        bounce_max_horizontal_impulse_ratio: float = 1.5,
        hit_min_direction_change_deg: float = 35.0,
        hit_min_impulse_px_per_second: float = 300.0,
        discontinuity_hit_max_speed_px_per_second: float = 3200.0,
        event_cooldown_ms: float = 160.0,
        post_hit_bounce_suppression_ms: float = 180.0,
        rally_state_timeout_ms: float = 5000.0,
        player_contact_margin_ratio: float = 0.20,
        player_reach_margin_ratio: float = 0.35,
        player_foot_band_ratio: float = 0.18,
        net_y_ft: float = 22.0,
        net_deadband_ft: float = 0.75,
    ):
        self.enabled = bool(enabled)
        self.hit_flash_frames = max(1, int(hit_flash_frames))
        self.ground_min_hold_frames = max(1, int(ground_min_hold_frames))
        self.rise_confirm_frames = max(1, int(rise_confirm_frames))
        self.rise_min_speed_px_per_second = max(
            0.0,
            float(rise_min_speed_px_per_second),
        )
        self.kinematic_enabled = bool(kinematic_enabled)
        self.max_observation_gap_s = max(
            0.0,
            float(max_observation_gap_ms) / 1000.0,
        )
        self.bounce_min_downward_speed_px_per_second = max(
            0.0,
            float(bounce_min_downward_speed_px_per_second),
        )
        self.bounce_min_upward_speed_px_per_second = max(
            0.0,
            float(bounce_min_upward_speed_px_per_second),
        )
        self.bounce_max_horizontal_impulse_ratio = max(
            0.0,
            float(bounce_max_horizontal_impulse_ratio),
        )
        self.hit_min_direction_change_deg = min(
            180.0,
            max(0.0, float(hit_min_direction_change_deg)),
        )
        self.hit_min_impulse_px_per_second = max(
            0.0,
            float(hit_min_impulse_px_per_second),
        )
        self.discontinuity_hit_max_speed_px_per_second = max(
            self.hit_min_impulse_px_per_second,
            float(discontinuity_hit_max_speed_px_per_second),
        )
        self.event_cooldown_s = max(
            0.0,
            float(event_cooldown_ms) / 1000.0,
        )
        self.post_hit_bounce_suppression_s = max(
            0.0,
            float(post_hit_bounce_suppression_ms) / 1000.0,
        )
        self.rally_state_timeout_s = max(
            0.0,
            float(rally_state_timeout_ms) / 1000.0,
        )
        self.player_contact_margin_ratio = max(
            0.0,
            float(player_contact_margin_ratio),
        )
        self.player_reach_margin_ratio = max(
            self.player_contact_margin_ratio,
            float(player_reach_margin_ratio),
        )
        self.player_foot_band_ratio = min(
            0.5,
            max(0.0, float(player_foot_band_ratio)),
        )
        self.net_y_ft = float(net_y_ft)
        self.net_deadband_ft = max(0.0, float(net_deadband_ft))
        self._previous_counters = {
            side: {"bounce_recoveries": 0, "impact_recoveries": 0}
            for side in ("left", "right")
        }
        self._track_identity: tuple[str, int | None] | None = None
        self._motion_history: deque[_MotionSample] = deque(maxlen=4)
        self._update_index = -1
        self._ground_state: str | None = None
        self._ground_age_frames = 0
        self._rising_frames = 0
        self._hit_frames_remaining = 0
        self._bounce_index_since_last_hit = 0
        self._last_bounce_half: str | None = None
        self._last_bounce_time_s = float("-inf")
        self._last_event_time_s = float("-inf")
        self._last_hit_time_s = float("-inf")
        self._last_available_time_s = float("-inf")

    def reset(self) -> None:
        self._track_identity = None
        self._motion_history.clear()
        self._clear_latched_state()
        self._bounce_index_since_last_hit = 0
        self._last_bounce_half = None
        self._last_bounce_time_s = float("-inf")
        self._last_event_time_s = float("-inf")
        self._last_hit_time_s = float("-inf")
        self._last_available_time_s = float("-inf")

    def update(
        self,
        projection: ProjectionResult,
        *,
        track,
        active_side: str | None,
        local_track_id: int | None,
        tracker_diagnostics: dict[str, dict],
        frame_scale_overrides: dict[str, float],
        discontinuity_reason: str | None = None,
        timestamp_s: float | None = None,
        frame_index: int | None = None,
        eligible_players: list[Any] | None = None,
    ) -> CourtEventResult:
        deltas = self._diagnostic_deltas(tracker_diagnostics)
        self._update_index += 1
        now_s = (
            float(timestamp_s)
            if timestamp_s is not None
            else self._update_index / 50.0
        )
        current_frame_index = (
            int(frame_index) if frame_index is not None else self._update_index
        )
        if not self.enabled:
            return CourtEventResult(
                phase="unknown",
                display_state="unknown",
                display_text_zh="状态未知",
            )

        result = CourtEventResult(
            bounce_index_since_last_hit=self._bounce_index_since_last_hit,
        )
        if (
            active_side not in ("left", "right")
            or track is None
            or not projection.projection_valid
        ):
            if (
                isfinite(self._last_available_time_s)
                and self.rally_state_timeout_s > 0.0
                and now_s - self._last_available_time_s
                > self.rally_state_timeout_s
            ):
                self.reset()
                result.bounce_index_since_last_hit = 0
                result.warnings.append(
                    "event_state_reset_after_unavailable_timeout"
                )
            result.phase = "unknown"
            result.display_color = "fluorescent_green"
            result.warnings.append("event_input_unavailable")
            if track is None or active_side not in ("left", "right"):
                result.display_state = "not_visible"
                result.display_text_zh = "消失在屏幕"
            else:
                result.display_state = "projection_unavailable"
                result.display_text_zh = "投影不可用"
            return result

        self._last_available_time_s = now_s
        previous_identity = self._track_identity
        previous_motion_history = tuple(self._motion_history)
        identity = (active_side, local_track_id)
        identity_changed = (
            self._track_identity is not None
            and identity != self._track_identity
        )
        reset_for_discontinuity = discontinuity_reason in {
            "local_track_change",
            "physical_discontinuity",
            "camera_side_switch",
        }
        if reset_for_discontinuity or identity_changed:
            self._motion_history.clear()
            self._clear_latched_state()
            if discontinuity_reason == "physical_discontinuity":
                self._bounce_index_since_last_hit = 0
                self._last_bounce_half = None
                self._last_bounce_time_s = float("-inf")
            result.warnings.append("event_state_reset_on_track_discontinuity")
        self._track_identity = identity

        frame_scale = max(
            float(frame_scale_overrides.get(active_side, 1.0)),
            1e-6,
        )
        current_sample = self._build_sample(
            projection,
            now_s,
            current_frame_index,
            eligible_players or [],
        )
        if current_sample is not None:
            self._append_motion_sample(current_sample)

        discontinuity_hit = self._discontinuity_hit_candidate(
            previous_identity=previous_identity,
            current_identity=identity,
            previous_history=previous_motion_history,
            current_sample=current_sample,
            track=track,
            frame_scale=frame_scale,
            discontinuity_reason=discontinuity_reason,
        )

        side_delta = deltas[active_side]
        r9_hit = bool(
            side_delta["impact_recoveries"]
            and getattr(track, "status", None) == "observed"
        )
        r9_bounce = bool(
            side_delta["bounce_recoveries"]
            and getattr(track, "status", None) == "observed"
        )
        kinematic = self._kinematic_candidate(frame_scale)

        candidate_kind: str | None = None
        candidate_sample = current_sample
        candidate_evidence: list[str] = []
        candidate_warnings: list[str] = []
        candidate_metrics: dict[str, float | bool | str] = {}

        if r9_hit:
            candidate_kind = "hit"
            candidate_evidence.extend(
                [
                    "r9_impact_recovery_counter_increment",
                    "existing_eligible_player_contact_gate",
                ]
            )
            candidate_warnings.append("tracker_diagnostic_not_track_scoped")
        elif r9_bounce and current_sample is not None:
            if current_sample.player_relation == "body_contact":
                candidate_kind = "hit"
                candidate_evidence.extend(
                    [
                        "r9_bounce_recovery_counter_increment",
                        "existing_eligible_player_body_proximity",
                        "r9_bounce_reclassified_as_hit",
                    ]
                )
                candidate_warnings.append(
                    "tracker_diagnostic_not_track_scoped"
                )
            else:
                candidate_kind = "bounce"
                candidate_evidence.append(
                    "r9_bounce_recovery_counter_increment"
                )
                candidate_warnings.append(
                    "tracker_diagnostic_not_track_scoped"
                )

        if candidate_kind is None and discontinuity_hit is not None:
            candidate_kind = "hit"
            candidate_sample = discontinuity_hit.contact
            candidate_evidence.extend(
                [
                    "existing_eligible_player_contact_or_reach_proximity",
                    discontinuity_hit.source,
                ]
            )
            if discontinuity_hit.source == "same_side_track_change_reversal":
                candidate_evidence.extend(
                    [
                        "short_gap_local_track_identity_change",
                        "cross_track_image_velocity_reversal",
                    ]
                )
            else:
                candidate_evidence.extend(
                    [
                        "camera_side_switch_observation",
                        "fast_new_side_observed_track",
                    ]
                )
            candidate_metrics = {
                "post_vx_px_per_second": round(
                    discontinuity_hit.post_velocity[0],
                    3,
                ),
                "post_vy_px_per_second": round(
                    discontinuity_hit.post_velocity[1],
                    3,
                ),
                "impulse_speed_px_per_second": round(
                    discontinuity_hit.impulse_speed_px_per_second,
                    3,
                ),
                "player_relation": discontinuity_hit.contact.player_relation,
                "discontinuity_source": discontinuity_hit.source,
            }
            if discontinuity_hit.pre_velocity is not None:
                candidate_metrics.update(
                    {
                        "pre_vx_px_per_second": round(
                            discontinuity_hit.pre_velocity[0],
                            3,
                        ),
                        "pre_vy_px_per_second": round(
                            discontinuity_hit.pre_velocity[1],
                            3,
                        ),
                    }
                )
            if discontinuity_hit.direction_change_deg is not None:
                candidate_metrics["direction_change_deg"] = round(
                    discontinuity_hit.direction_change_deg,
                    3,
                )

        if candidate_kind is None and kinematic is not None:
            candidate_kind = kinematic.kind
            candidate_sample = kinematic.contact
            candidate_evidence.append("multi_frame_observed_image_kinematics")
            if kinematic.kind == "bounce":
                candidate_evidence.extend(
                    [
                        "vertical_down_to_up_reversal",
                        "vertical_impulse_dominates_horizontal_impulse",
                    ]
                )
            else:
                candidate_evidence.extend(
                    [
                        "existing_eligible_player_contact_proximity",
                        "sharp_image_plane_velocity_change",
                    ]
                )
            candidate_metrics = {
                "pre_vx_px_per_second": round(
                    kinematic.pre_velocity[0],
                    3,
                ),
                "pre_vy_px_per_second": round(
                    kinematic.pre_velocity[1],
                    3,
                ),
                "post_vx_px_per_second": round(
                    kinematic.post_velocity[0],
                    3,
                ),
                "post_vy_px_per_second": round(
                    kinematic.post_velocity[1],
                    3,
                ),
                "direction_change_deg": round(
                    kinematic.direction_change_deg,
                    3,
                ),
                "impulse_speed_px_per_second": round(
                    kinematic.impulse_speed_px_per_second,
                    3,
                ),
                "horizontal_impulse_ratio": round(
                    kinematic.horizontal_impulse_ratio,
                    6,
                ),
                "player_relation": kinematic.contact.player_relation,
            }

        candidate_kind = self._apply_event_suppression(
            candidate_kind,
            now_s,
            result,
        )
        if candidate_kind == "hit" and candidate_sample is not None:
            self._register_hit(
                result,
                candidate_sample,
                now_s,
                candidate_evidence,
                candidate_warnings,
                candidate_metrics,
            )
        elif candidate_kind == "bounce" and candidate_sample is not None:
            self._register_bounce(
                result,
                candidate_sample,
                now_s,
                candidate_evidence,
                candidate_warnings,
                candidate_metrics,
            )

        if self._hit_frames_remaining > 0:
            result.phase = "hit_flash"
            result.display_color = "purple"
            result.display_state = "hit"
            result.display_text_zh = "击球"
            self._hit_frames_remaining -= 1
        elif self._ground_state is not None:
            self._ground_age_frames += 1
            if self._is_confirmed_rising(track, frame_scale):
                self._rising_frames += 1
            else:
                self._rising_frames = 0

            if (
                self._ground_age_frames > self.ground_min_hold_frames
                and self._rising_frames >= self.rise_confirm_frames
            ):
                self._ground_state = None
                self._ground_age_frames = 0
                self._rising_frames = 0
                result.phase = "airborne"
                result.display_color = "fluorescent_green"
                result.display_state = "airborne"
                result.display_text_zh = "飞行"
            elif self._ground_state == "landed_outside":
                result.phase = "ground_contact_hold"
                result.display_color = "red"
                result.display_state = "out_of_bounds"
                result.display_text_zh = "出界"
                result.fault_candidate = True
                result.fault_reasons.append("out_of_bounds")
            elif self._bounce_index_since_last_hit >= 2:
                result.phase = "ground_contact_hold"
                result.display_color = "yellow"
                result.display_state = "second_bounce"
                result.display_text_zh = "二弹"
                result.fault_candidate = True
                result.fault_reasons.append("second_bounce")
            else:
                result.phase = "ground_contact_hold"
                result.display_color = "yellow"
                result.display_state = "bounce"
                result.display_text_zh = "落地"
        elif getattr(track, "status", None) == "predicted":
            result.display_state = "predicted"
            result.display_text_zh = "短时预测"

        result.bounce_index_since_last_hit = self._bounce_index_since_last_hit
        result.candidate = bool(result.events)
        return result

    def _build_sample(
        self,
        projection: ProjectionResult,
        timestamp_s: float,
        frame_index: int,
        eligible_players: list[Any],
    ) -> _MotionSample | None:
        if (
            not projection.observed
            or projection.image_xy is None
            or len(projection.image_xy) < 2
        ):
            return None
        image_xy = (
            float(projection.image_xy[0]),
            float(projection.image_xy[1]),
        )
        court_xy = (
            (
                float(projection.ball_court_xy[0]),
                float(projection.ball_court_xy[1]),
            )
            if projection.ball_court_xy is not None
            and len(projection.ball_court_xy) >= 2
            else None
        )
        return _MotionSample(
            timestamp_s=timestamp_s,
            frame_index=frame_index,
            image_xy=image_xy,
            court_xy=court_xy,
            inside_court=projection.inside_court,
            player_relation=self._player_relation(
                image_xy,
                eligible_players,
            ),
        )

    def _append_motion_sample(self, sample: _MotionSample) -> None:
        if (
            self._motion_history
            and (
                sample.timestamp_s
                <= self._motion_history[-1].timestamp_s
                or sample.timestamp_s
                - self._motion_history[-1].timestamp_s
                > self.max_observation_gap_s
            )
        ):
            self._motion_history.clear()
        self._motion_history.append(sample)

    def _kinematic_candidate(
        self,
        frame_scale: float,
    ) -> _KinematicCandidate | None:
        if not self.kinematic_enabled or len(self._motion_history) < 4:
            return None
        start, contact, after_1, after_2 = self._motion_history
        pre_dt = contact.timestamp_s - start.timestamp_s
        post_dt = after_2.timestamp_s - contact.timestamp_s
        if pre_dt <= 0.0 or post_dt <= 0.0:
            return None
        pre_velocity = (
            (contact.image_xy[0] - start.image_xy[0]) / pre_dt,
            (contact.image_xy[1] - start.image_xy[1]) / pre_dt,
        )
        post_velocity = (
            (after_2.image_xy[0] - contact.image_xy[0]) / post_dt,
            (after_2.image_xy[1] - contact.image_xy[1]) / post_dt,
        )
        delta_vx = post_velocity[0] - pre_velocity[0]
        delta_vy = post_velocity[1] - pre_velocity[1]
        impulse = hypot(delta_vx, delta_vy)
        horizontal_ratio = abs(delta_vx) / max(abs(delta_vy), 1.0)
        direction_change = self._direction_change_deg(
            pre_velocity,
            post_velocity,
        )
        vertical_reversal = (
            pre_velocity[1]
            >= self.bounce_min_downward_speed_px_per_second * frame_scale
            and post_velocity[1]
            <= -self.bounce_min_upward_speed_px_per_second * frame_scale
            and after_1.image_xy[1] < contact.image_xy[1]
            and after_2.image_xy[1] < after_1.image_xy[1]
        )
        hit_motion = (
            contact.player_relation in {
                "body_contact",
                "extended_player_reach",
            }
            and impulse
            >= self.hit_min_impulse_px_per_second * frame_scale
            and (
                direction_change >= self.hit_min_direction_change_deg
                or abs(delta_vx)
                >= self.hit_min_impulse_px_per_second * frame_scale
            )
        )
        if hit_motion:
            kind = "hit"
        elif (
            vertical_reversal
            and horizontal_ratio
            <= self.bounce_max_horizontal_impulse_ratio
        ):
            kind = "bounce"
        else:
            return None
        return _KinematicCandidate(
            kind=kind,
            contact=contact,
            pre_velocity=pre_velocity,
            post_velocity=post_velocity,
            direction_change_deg=direction_change,
            impulse_speed_px_per_second=impulse,
            horizontal_impulse_ratio=horizontal_ratio,
        )

    def _discontinuity_hit_candidate(
        self,
        *,
        previous_identity: tuple[str, int | None] | None,
        current_identity: tuple[str, int | None],
        previous_history: tuple[_MotionSample, ...],
        current_sample: _MotionSample | None,
        track,
        frame_scale: float,
        discontinuity_reason: str | None,
    ) -> _DiscontinuityHitCandidate | None:
        if (
            not self.kinematic_enabled
            or current_sample is None
            or getattr(track, "status", None) != "observed"
            or current_sample.player_relation
            not in {"body_contact", "extended_player_reach"}
            or previous_identity is None
        ):
            return None

        same_side_track_change = (
            previous_identity[0] == current_identity[0]
            and previous_identity[1] != current_identity[1]
        )
        if same_side_track_change and len(previous_history) >= 2:
            before = previous_history[-2]
            contact_before = previous_history[-1]
            pre_dt = contact_before.timestamp_s - before.timestamp_s
            post_dt = current_sample.timestamp_s - contact_before.timestamp_s
            if (
                pre_dt <= 0.0
                or post_dt <= 0.0
                or post_dt > self.max_observation_gap_s
            ):
                return None
            pre_velocity = (
                (
                    contact_before.image_xy[0] - before.image_xy[0]
                )
                / pre_dt,
                (
                    contact_before.image_xy[1] - before.image_xy[1]
                )
                / pre_dt,
            )
            post_velocity = (
                (
                    current_sample.image_xy[0] - contact_before.image_xy[0]
                )
                / post_dt,
                (
                    current_sample.image_xy[1] - contact_before.image_xy[1]
                )
                / post_dt,
            )
            direction_change = self._direction_change_deg(
                pre_velocity,
                post_velocity,
            )
            impulse = hypot(
                post_velocity[0] - pre_velocity[0],
                post_velocity[1] - pre_velocity[1],
            )
            post_speed = hypot(*post_velocity)
            if (
                direction_change < self.hit_min_direction_change_deg
                or impulse
                < self.hit_min_impulse_px_per_second * frame_scale
                or post_speed
                > self.discontinuity_hit_max_speed_px_per_second
                * frame_scale
            ):
                return None
            return _DiscontinuityHitCandidate(
                source="same_side_track_change_reversal",
                contact=current_sample,
                pre_velocity=pre_velocity,
                post_velocity=post_velocity,
                direction_change_deg=direction_change,
                impulse_speed_px_per_second=impulse,
            )

        camera_side_switch = (
            previous_identity[0] != current_identity[0]
            and discontinuity_reason == "camera_side_switch"
        )
        velocity = getattr(track, "velocity", None)
        if (
            not camera_side_switch
            or velocity is None
            or len(velocity) < 2
        ):
            return None
        post_velocity = (float(velocity[0]), float(velocity[1]))
        speed = hypot(*post_velocity)
        if (
            speed < self.hit_min_impulse_px_per_second * frame_scale
            or speed
            > self.discontinuity_hit_max_speed_px_per_second * frame_scale
        ):
            return None
        return _DiscontinuityHitCandidate(
            source="camera_side_switch_extended_reach",
            contact=current_sample,
            pre_velocity=None,
            post_velocity=post_velocity,
            direction_change_deg=None,
            impulse_speed_px_per_second=speed,
        )

    def _apply_event_suppression(
        self,
        candidate_kind: str | None,
        now_s: float,
        result: CourtEventResult,
    ) -> str | None:
        if candidate_kind is None:
            return None
        if (
            candidate_kind == "bounce"
            and now_s - self._last_hit_time_s
            < self.post_hit_bounce_suppression_s
        ):
            result.warnings.append("bounce_candidate_suppressed_post_hit")
            return None
        if now_s - self._last_event_time_s < self.event_cooldown_s:
            result.warnings.append("event_candidate_suppressed_by_cooldown")
            return None
        return candidate_kind

    def _register_hit(
        self,
        result: CourtEventResult,
        contact: _MotionSample,
        now_s: float,
        evidence: list[str],
        warnings: list[str],
        metrics: dict[str, float | bool | str],
    ) -> None:
        self._hit_frames_remaining = self.hit_flash_frames
        self._ground_state = None
        self._ground_age_frames = 0
        self._rising_frames = 0
        self._bounce_index_since_last_hit = 0
        self._last_bounce_half = None
        self._last_bounce_time_s = float("-inf")
        self._last_event_time_s = now_s
        self._last_hit_time_s = now_s
        result.events.append("paddle_hit_candidate")
        result.evidence.extend(evidence)
        result.warnings.extend(warnings)
        result.warnings.append("paddle_not_directly_detected")
        self._attach_contact(result, contact, metrics)

    def _register_bounce(
        self,
        result: CourtEventResult,
        contact: _MotionSample,
        now_s: float,
        evidence: list[str],
        warnings: list[str],
        metrics: dict[str, float | bool | str],
    ) -> None:
        contact_half = self._contact_physical_half(contact)
        stale_rally_state = (
            contact_half is not None
            and self._last_bounce_half == contact_half
            and self.rally_state_timeout_s > 0.0
            and now_s - self._last_bounce_time_s
            > self.rally_state_timeout_s
        )
        if (
            contact_half is None
            or self._last_bounce_half != contact_half
            or stale_rally_state
        ):
            self._bounce_index_since_last_hit = 0
        if stale_rally_state:
            result.warnings.append("rally_state_reset_after_timeout")
        self._bounce_index_since_last_hit += 1
        self._last_bounce_half = contact_half
        self._last_bounce_time_s = now_s
        self._ground_state = (
            "landed_outside"
            if contact.inside_court is False
            else "landed_inside"
        )
        self._ground_age_frames = 0
        self._rising_frames = 0
        self._last_event_time_s = now_s
        result.events.append("bounce_candidate")
        if contact.inside_court is False:
            result.events.append("out_of_bounds_bounce_candidate")
            result.warnings.append(
                "out_of_bounds_requires_calibration_validation"
            )
        elif self._bounce_index_since_last_hit >= 2:
            result.events.append("second_bounce_candidate")
        result.evidence.extend(evidence)
        result.evidence.append(
            "ground_plane_projection_at_contact_candidate"
        )
        result.warnings.extend(warnings)
        result.warnings.append("image_plane_kinematics_only")
        self._attach_contact(result, contact, metrics)

    def _contact_physical_half(self, contact: _MotionSample) -> str | None:
        if contact.court_xy is None or len(contact.court_xy) < 2:
            return None
        y_ft = float(contact.court_xy[1])
        if not isfinite(y_ft):
            return None
        if y_ft < self.net_y_ft - self.net_deadband_ft:
            return "low_y"
        if y_ft > self.net_y_ft + self.net_deadband_ft:
            return "high_y"
        return None

    @staticmethod
    def _attach_contact(
        result: CourtEventResult,
        contact: _MotionSample,
        metrics: dict[str, float | bool | str],
    ) -> None:
        result.contact_frame_index = contact.frame_index
        result.contact_timestamp_s = contact.timestamp_s
        result.contact_image_xy = list(contact.image_xy)
        result.contact_court_xy = (
            list(contact.court_xy) if contact.court_xy is not None else None
        )
        result.contact_inside_court = contact.inside_court
        result.metrics.update(metrics)
        result.metrics.setdefault("player_relation", contact.player_relation)

    def _diagnostic_deltas(
        self,
        tracker_diagnostics: dict[str, dict],
    ) -> dict[str, dict[str, bool]]:
        deltas: dict[str, dict[str, bool]] = {}
        for side in ("left", "right"):
            current_diagnostics = tracker_diagnostics.get(side, {})
            deltas[side] = {}
            for name in ("bounce_recoveries", "impact_recoveries"):
                current = max(0, int(current_diagnostics.get(name, 0)))
                previous = self._previous_counters[side][name]
                deltas[side][name] = current > previous
                self._previous_counters[side][name] = current
        return deltas

    def _player_relation(
        self,
        point: tuple[float, float],
        eligible_players: list[Any],
    ) -> str:
        relation = "none"
        for player in eligible_players:
            eligible = self._player_value(
                player,
                "eligible_player",
                False,
            )
            bbox = self._player_value(player, "bbox", None)
            if not eligible or bbox is None or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            height = max(1.0, y2 - y1)
            reach_margin = height * self.player_reach_margin_ratio
            if not (
                x1 - reach_margin <= point[0] <= x2 + reach_margin
                and y1 - reach_margin
                <= point[1]
                <= y2 + 0.1 * height
            ):
                continue
            margin = height * self.player_contact_margin_ratio
            inside_body_contact = (
                x1 - margin <= point[0] <= x2 + margin
                and y1 - margin <= point[1] <= y2 + 0.1 * height
            )
            if not inside_body_contact:
                relation = "extended_player_reach"
                continue
            if point[1] >= y2 - self.player_foot_band_ratio * height:
                relation = "foot_contact"
            else:
                return "body_contact"
        return relation

    @staticmethod
    def _player_value(
        player: Any,
        name: str,
        default: Any,
    ) -> Any:
        if isinstance(player, dict):
            return player.get(name, default)
        return getattr(player, name, default)

    def _is_confirmed_rising(self, track, frame_scale: float) -> bool:
        velocity = getattr(track, "velocity", None)
        if (
            getattr(track, "status", None) != "observed"
            or velocity is None
            or len(velocity) < 2
        ):
            return False
        threshold = self.rise_min_speed_px_per_second * frame_scale
        return float(velocity[1]) <= -threshold

    @staticmethod
    def _direction_change_deg(
        before: tuple[float, float],
        after: tuple[float, float],
    ) -> float:
        denominator = hypot(*before) * hypot(*after)
        if denominator <= 1e-9:
            return 0.0
        cosine = (
            before[0] * after[0] + before[1] * after[1]
        ) / denominator
        return degrees(acos(min(1.0, max(-1.0, cosine))))

    def _clear_latched_state(self) -> None:
        self._ground_state = None
        self._ground_age_frames = 0
        self._rising_frames = 0
        self._hit_frames_remaining = 0
