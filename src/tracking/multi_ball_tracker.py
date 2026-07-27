"""Lightweight multi-pickleball tracking with short-term motion prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .motion_models import ConstantAccelerationKalman, ConstantVelocityKalman
from .types import BallDetection, BallTrack


@dataclass
class _TrackState:
    track_id: int
    kalman: ConstantVelocityKalman | ConstantAccelerationKalman
    bbox_width: float
    bbox_height: float
    confidence: float
    age: int = 1
    hits: int = 1
    missing_frames: int = 0
    observed_this_frame: bool = True
    last_observed_center: tuple[float, float] = (0.0, 0.0)
    stationary_anchor_center: tuple[float, float] = (0.0, 0.0)
    stationary_anchor_frames: int = 0
    raw_stationary_anchor_center: tuple[float, float] = (0.0, 0.0)
    raw_stationary_anchor_frames: int = 0
    raw_last_observed_center: tuple[float, float] = (0.0, 0.0)
    motion_confirmed: bool = False
    stationary_frames: int = 0
    missing_time_s: float = 0.0
    last_observed_time_s: float = 0.0
    stationary_anchor_time_s: float = 0.0
    raw_stationary_anchor_time_s: float = 0.0
    stationary_time_s: float = 0.0
    age_time_s: float = 0.0
    last_observed_velocity: tuple[float, float] = (0.0, 0.0)
    last_observation_source: str = "unknown"
    bounce_recovery_until_s: float = -1.0

    @property
    def center(self) -> tuple[float, float]:
        return self.kalman.position

    @property
    def bbox(self) -> list[float]:
        cx, cy = self.center
        return [
            cx - self.bbox_width / 2,
            cy - self.bbox_height / 2,
            cx + self.bbox_width / 2,
            cy + self.bbox_height / 2,
        ]


class MultiBallTracker:
    """Associate every ball detection to an independent motion track.

    High-confidence detections create and update tracks. Lower-confidence
    detections may recover an existing track but never create a new one. When a
    detector misses briefly, a constant-velocity Kalman prediction fills the
    trajectory until the configured prediction time is reached.
    """

    def __init__(
        self,
        high_conf: float = 0.25,
        low_conf: float = 0.08,
        min_hits: int = 2,
        max_predict_frames: int = 8,
        max_missing_frames: int = 20,
        max_prediction_ms: Optional[float] = None,
        max_missing_ms: Optional[float] = None,
        base_gate_px: float = 70.0,
        max_gate_px: float = 280.0,
        gate_growth_px: float = 20.0,
        gate_growth_px_per_second: Optional[float] = None,
        velocity_gate_scale: float = 1.5,
        tentative_gate_px: float = 150.0,
        max_speed_px_per_frame: float = 0.0,
        max_speed_px_per_second: Optional[float] = None,
        reference_frame_width: float = 1280.0,
        frame_scale_override: Optional[float] = None,
        default_fps: float = 30.0,
        decay_reference_fps: float = 30.0,
        require_motion_confirmation: bool = False,
        motion_threshold_px: float = 5.0,
        max_stationary_frames: int = 0,
        max_stationary_ms: Optional[float] = None,
        unconfirmed_max_frames: int = 30,
        unconfirmed_max_ms: Optional[float] = None,
        max_output_tracks: int = 0,
        observation_first_output: bool = False,
        impact_recovery_gate_px: float = 260.0,
        impact_recovery_min_speed_px: float = 15.0,
        impact_recovery_min_speed_px_per_second: Optional[float] = None,
        impact_recovery_min_missing_ms: float = 0.0,
        impact_recovery_max_missing_ms: Optional[float] = None,
        require_contact_for_impact_recovery: bool = False,
        contact_margin_ratio: float = 0.20,
        bounce_recovery_enabled: bool = False,
        bounce_recovery_sources: Optional[list[str]] = None,
        bounce_recovery_min_downward_speed_px_per_second: float = 0.0,
        bounce_recovery_min_upward_speed_px_per_second: float = 0.0,
        bounce_recovery_min_horizontal_speed_px_per_second: float = 0.0,
        bounce_recovery_max_displacement_px: float = 35.0,
        bounce_recovery_max_missing_ms: float = 80.0,
        bounce_recovery_require_no_contact: bool = True,
        primary_continuity_recovery_enabled: bool = False,
        primary_observation_sources: Optional[list[str]] = None,
        primary_continuity_gate_px: float = 55.0,
        max_flight_direction_change_deg: float = 0.0,
        direction_gate_min_speed_px_per_second: float = 0.0,
        direction_gate_min_hits: int = 3,
        prediction_min_confidence: float = 0.0,
        fast_prediction_speed_px: float = 0.0,
        fast_prediction_speed_px_per_second: Optional[float] = None,
        fast_max_predict_frames: int = 3,
        fast_max_prediction_ms: Optional[float] = None,
        prediction_velocity_decay: float = 1.0,
        min_motion_speed_px: float = 0.0,
        min_motion_speed_px_per_second: Optional[float] = None,
        confidence_decay: float = 0.82,
        motion_model: str = "constant_velocity",
        process_noise: float = 1200.0,
        acceleration_process_noise: Optional[float] = None,
        measurement_noise: float = 4.0,
        constant_acceleration_min_observations: int = 4,
        max_acceleration_px_per_second2: float = 0.0,
        acceleration_decay: float = 0.8,
        use_nis_gate: bool = False,
        nis_gate_threshold: float = 9.21,
        nis_gate_min_hits: int = 3,
        max_observed_acceleration_px_per_second2: float = 0.0,
        acceleration_gate_min_hits: int = 3,
        continuous_prediction_horizon: bool = False,
        max_tracks: int = 16,
        emit_tentative: bool = True,
    ):
        if not 0.0 <= low_conf <= high_conf <= 1.0:
            raise ValueError("Require 0 <= low_conf <= high_conf <= 1")
        if max_predict_frames < 0 or max_missing_frames < max_predict_frames:
            raise ValueError("max_missing_frames must be >= max_predict_frames >= 0")
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.min_hits = max(1, int(min_hits))
        self.default_fps = max(1.0, float(default_fps))
        self.default_dt_s = 1.0 / self.default_fps
        self.decay_reference_dt_s = 1.0 / max(1.0, float(decay_reference_fps))
        self.max_predict_frames = int(max_predict_frames)
        self.max_missing_frames = int(max_missing_frames)
        self.max_prediction_s = (
            self.max_predict_frames / self.default_fps
            if max_prediction_ms is None
            else max(0.0, float(max_prediction_ms) / 1000.0)
        )
        self.max_missing_s = (
            self.max_missing_frames / self.default_fps
            if max_missing_ms is None
            else max(0.0, float(max_missing_ms) / 1000.0)
        )
        if self.max_missing_s < self.max_prediction_s:
            raise ValueError("max_missing_ms must be >= max_prediction_ms >= 0")
        self.base_gate_px = float(base_gate_px)
        self.max_gate_px = float(max_gate_px)
        self.gate_growth_px = float(gate_growth_px)
        self.gate_growth_px_per_second = (
            self.gate_growth_px * self.default_fps
            if gate_growth_px_per_second is None
            else max(0.0, float(gate_growth_px_per_second))
        )
        self.velocity_gate_scale = float(velocity_gate_scale)
        self.tentative_gate_px = float(tentative_gate_px)
        self.max_speed_px_per_frame = float(max_speed_px_per_frame)
        self.max_speed_px_per_second = (
            self.max_speed_px_per_frame * self.default_fps
            if max_speed_px_per_second is None
            else max(0.0, float(max_speed_px_per_second))
        )
        self.reference_frame_width = max(1.0, float(reference_frame_width))
        self.frame_scale_override = (
            None
            if frame_scale_override is None
            else max(0.25, float(frame_scale_override))
        )
        self.require_motion_confirmation = bool(require_motion_confirmation)
        self.motion_threshold_px = max(0.0, float(motion_threshold_px))
        self.max_stationary_frames = max(0, int(max_stationary_frames))
        self.max_stationary_s = (
            self.max_stationary_frames / self.default_fps
            if max_stationary_ms is None
            else max(0.0, float(max_stationary_ms) / 1000.0)
        )
        self.unconfirmed_max_frames = max(1, int(unconfirmed_max_frames))
        self.unconfirmed_max_s = (
            self.unconfirmed_max_frames / self.default_fps
            if unconfirmed_max_ms is None
            else max(0.0, float(unconfirmed_max_ms) / 1000.0)
        )
        self.max_output_tracks = max(0, int(max_output_tracks))
        self.observation_first_output = bool(observation_first_output)
        self.impact_recovery_gate_px = max(0.0, float(impact_recovery_gate_px))
        self.impact_recovery_min_speed_px = max(0.0, float(impact_recovery_min_speed_px))
        self.impact_recovery_min_speed_px_per_second = (
            self.impact_recovery_min_speed_px * self.default_fps
            if impact_recovery_min_speed_px_per_second is None
            else max(0.0, float(impact_recovery_min_speed_px_per_second))
        )
        self.impact_recovery_min_missing_s = max(
            0.0, float(impact_recovery_min_missing_ms) / 1000.0
        )
        self.impact_recovery_max_missing_s = (
            None
            if impact_recovery_max_missing_ms is None
            else max(0.0, float(impact_recovery_max_missing_ms) / 1000.0)
        )
        if (
            self.impact_recovery_max_missing_s is not None
            and self.impact_recovery_max_missing_s
            < self.impact_recovery_min_missing_s
        ):
            raise ValueError(
                "impact_recovery_max_missing_ms must be >= "
                "impact_recovery_min_missing_ms"
            )
        self.require_contact_for_impact_recovery = bool(
            require_contact_for_impact_recovery
        )
        self.contact_margin_ratio = max(0.0, float(contact_margin_ratio))
        self.bounce_recovery_enabled = bool(bounce_recovery_enabled)
        self.bounce_recovery_sources = frozenset(
            str(source).strip().lower()
            for source in (bounce_recovery_sources or [])
            if str(source).strip()
        )
        self.bounce_recovery_min_downward_speed_px_per_second = max(
            0.0, float(bounce_recovery_min_downward_speed_px_per_second)
        )
        self.bounce_recovery_min_upward_speed_px_per_second = max(
            0.0, float(bounce_recovery_min_upward_speed_px_per_second)
        )
        self.bounce_recovery_min_horizontal_speed_px_per_second = max(
            0.0, float(bounce_recovery_min_horizontal_speed_px_per_second)
        )
        self.bounce_recovery_max_displacement_px = max(
            0.0, float(bounce_recovery_max_displacement_px)
        )
        self.bounce_recovery_max_missing_s = max(
            0.0, float(bounce_recovery_max_missing_ms) / 1000.0
        )
        self.bounce_recovery_require_no_contact = bool(
            bounce_recovery_require_no_contact
        )
        self.primary_continuity_recovery_enabled = bool(
            primary_continuity_recovery_enabled
        )
        self.primary_observation_sources = frozenset(
            str(source).strip().lower()
            for source in (primary_observation_sources or [])
            if str(source).strip()
        )
        self.primary_continuity_gate_px = max(
            0.0, float(primary_continuity_gate_px)
        )
        self.max_flight_direction_change_deg = min(
            180.0, max(0.0, float(max_flight_direction_change_deg))
        )
        self.direction_gate_min_speed_px_per_second = max(
            0.0, float(direction_gate_min_speed_px_per_second)
        )
        self.direction_gate_min_hits = max(2, int(direction_gate_min_hits))
        self.prediction_min_confidence = max(0.0, float(prediction_min_confidence))
        self.fast_prediction_speed_px = max(0.0, float(fast_prediction_speed_px))
        self.fast_prediction_speed_px_per_second = (
            self.fast_prediction_speed_px * self.default_fps
            if fast_prediction_speed_px_per_second is None
            else max(0.0, float(fast_prediction_speed_px_per_second))
        )
        self.fast_max_predict_frames = max(0, int(fast_max_predict_frames))
        self.fast_max_prediction_s = (
            self.fast_max_predict_frames / self.default_fps
            if fast_max_prediction_ms is None
            else max(0.0, float(fast_max_prediction_ms) / 1000.0)
        )
        self.prediction_velocity_decay = min(1.0, max(0.0, float(prediction_velocity_decay)))
        self.min_motion_speed_px = max(0.0, float(min_motion_speed_px))
        self.min_motion_speed_px_per_second = (
            self.min_motion_speed_px * self.default_fps
            if min_motion_speed_px_per_second is None
            else max(0.0, float(min_motion_speed_px_per_second))
        )
        self.confidence_decay = float(confidence_decay)
        self.motion_model = str(motion_model).strip().lower()
        if self.motion_model not in {"constant_velocity", "constant_acceleration"}:
            raise ValueError(
                "motion_model must be 'constant_velocity' or 'constant_acceleration'"
            )
        self.process_noise = float(process_noise)
        self.acceleration_process_noise = (
            self.process_noise
            if acceleration_process_noise is None
            else max(0.0, float(acceleration_process_noise))
        )
        self.measurement_noise = float(measurement_noise)
        self.constant_acceleration_min_observations = max(
            3, int(constant_acceleration_min_observations)
        )
        self.max_acceleration_px_per_second2 = max(
            0.0, float(max_acceleration_px_per_second2)
        )
        self.acceleration_decay = min(1.0, max(0.0, float(acceleration_decay)))
        self.use_nis_gate = bool(use_nis_gate)
        self.nis_gate_threshold = max(0.0, float(nis_gate_threshold))
        self.nis_gate_min_hits = max(2, int(nis_gate_min_hits))
        self.max_observed_acceleration_px_per_second2 = max(
            0.0, float(max_observed_acceleration_px_per_second2)
        )
        self.acceleration_gate_min_hits = max(3, int(acceleration_gate_min_hits))
        self.continuous_prediction_horizon = bool(continuous_prediction_horizon)
        self.max_tracks = max(1, int(max_tracks))
        self.emit_tentative = bool(emit_tentative)
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1
        self._last_diagnostics: dict = {}
        self._physical_rejections = 0
        self._preferred_track_id: Optional[int] = None
        self._impact_recoveries = 0
        self._bounce_recoveries = 0
        self._primary_continuity_recoveries = 0
        self._direction_gate_rejections = 0
        self._nis_gate_rejections = 0
        self._acceleration_gate_rejections = 0
        self._contact_gate_rejections = 0
        self._contact_zones: list[list[float]] = []
        self._last_timestamp_s: Optional[float] = None
        self._tracker_time_s = 0.0
        self._current_dt_s = self.default_dt_s

    @property
    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        detections: list[BallDetection],
        frame_width: int,
        frame_height: int,
        camera_motion: tuple[float, float] = (0.0, 0.0),
        timestamp_s: Optional[float] = None,
        contact_zones: Optional[list[list[float]]] = None,
    ) -> list[BallTrack]:
        dt_s = self._resolve_dt(timestamp_s)
        self._current_dt_s = dt_s
        self._tracker_time_s += dt_s
        self._contact_zones = [list(zone) for zone in (contact_zones or [])]
        valid = [d for d in detections if d.confidence >= self.low_conf]
        high = [d for d in valid if d.confidence >= self.high_conf]
        low = [d for d in valid if d.confidence < self.high_conf]

        camera_dx, camera_dy = (float(camera_motion[0]), float(camera_motion[1]))
        for track in self._tracks:
            self._shift_track_for_camera(track, camera_dx, camera_dy)
            if track.hits < self.constant_acceleration_min_observations:
                track.kalman.reset_acceleration()
            track.kalman.predict(dt_s)
            track.age += 1
            track.age_time_s += dt_s
            track.observed_this_frame = False

        unmatched_tracks = set(range(len(self._tracks)))
        self._physical_rejections = 0
        self._impact_recoveries = 0
        self._bounce_recoveries = 0
        self._primary_continuity_recoveries = 0
        self._direction_gate_rejections = 0
        self._nis_gate_rejections = 0
        self._acceleration_gate_rejections = 0
        self._contact_gate_rejections = 0
        high_matches, unmatched_high = self._associate(unmatched_tracks, high, frame_width)
        self._apply_matches(high_matches, high, frame_width)
        unmatched_tracks -= {track_index for track_index, _, _ in high_matches}

        low_matches, _ = self._associate(unmatched_tracks, low, frame_width)
        self._apply_matches(low_matches, low, frame_width)
        unmatched_tracks -= {track_index for track_index, _, _ in low_matches}

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.missing_frames += 1
            track.missing_time_s = max(
                0.0,
                self._tracker_time_s - track.last_observed_time_s,
            )
            decay_exponent = dt_s / self.decay_reference_dt_s
            track.confidence *= self.confidence_decay**decay_exponent
            track.kalman.damp_velocity(
                self.prediction_velocity_decay**decay_exponent
            )
            track.kalman.damp_acceleration(
                self.acceleration_decay**decay_exponent
            )

        stale_missing = {
            index for index, track in enumerate(self._tracks)
            if track.missing_time_s > self.max_missing_s
        }
        stale_unconfirmed = {
            index for index, track in enumerate(self._tracks)
            if self.require_motion_confirmation
            and not track.motion_confirmed
            and track.age_time_s > self.unconfirmed_max_s
        }
        removed = len(stale_missing | stale_unconfirmed)
        self._tracks = [
            track for index, track in enumerate(self._tracks)
            if index not in stale_missing and index not in stale_unconfirmed
        ]

        created = 0
        for detection_index in unmatched_high:
            if len(self._tracks) >= self.max_tracks:
                break
            self._create_track(high[detection_index])
            created += 1

        eligible_tracks = [
            track for track in self._tracks
            if track.missing_time_s <= self.max_prediction_s
            and self._prediction_is_reliable(track, frame_width, frame_height)
            and (self.emit_tentative or track.hits >= self.min_hits)
            and (not self.require_motion_confirmation or track.motion_confirmed)
            and (
                self.max_stationary_s == 0
                or track.stationary_time_s <= self.max_stationary_s
            )
        ]
        selected_tracks = self._select_output_tracks(eligible_tracks, frame_width)
        outputs = [
            self._to_output(track, frame_width, frame_height)
            for track in selected_tracks
        ]
        outputs.sort(key=lambda track: track.track_id or 0)
        self._last_diagnostics = {
            "input_detections": len(detections),
            "high_conf_detections": len(high),
            "low_conf_detections": len(low),
            "high_conf_matches": len(high_matches),
            "low_conf_recoveries": len(low_matches),
            "created_tracks": created,
            "removed_tracks": removed,
            "retained_tracks": len(self._tracks),
            "emitted_tracks": len(outputs),
            "eligible_tracks": len(eligible_tracks),
            "output_limited_tracks": len(eligible_tracks) - len(selected_tracks),
            "physical_gate_rejections": self._physical_rejections,
            "impact_recoveries": self._impact_recoveries,
            "bounce_recoveries": self._bounce_recoveries,
            "primary_continuity_recoveries": (
                self._primary_continuity_recoveries
            ),
            "direction_gate_rejections": self._direction_gate_rejections,
            "nis_gate_rejections": self._nis_gate_rejections,
            "acceleration_gate_rejections": self._acceleration_gate_rejections,
            "contact_gate_required": self.require_contact_for_impact_recovery,
            "contact_gate_rejections": self._contact_gate_rejections,
            "contact_zone_count": len(self._contact_zones),
            "motion_model": self.motion_model,
            "camera_motion_dx": camera_dx,
            "camera_motion_dy": camera_dy,
            "frame_dt_ms": round(dt_s * 1000.0, 3),
            "frame_scale": round(self._frame_scale(frame_width), 6),
            "frame_scale_overridden": self.frame_scale_override is not None,
            "tracker_time_s": round(self._tracker_time_s, 6),
            "motion_unconfirmed_tracks": sum(not track.motion_confirmed for track in self._tracks),
            "stationary_suppressed_tracks": sum(
                self.max_stationary_s > 0
                and track.stationary_time_s > self.max_stationary_s
                for track in self._tracks
            ),
            "track_states": [
                {
                    "track_id": track.track_id,
                    "center": [round(value, 3) for value in track.center],
                    "velocity": [round(value, 3) for value in track.kalman.velocity],
                    "acceleration": [
                        round(value, 3) for value in track.kalman.acceleration
                    ],
                    "missing_frames": track.missing_frames,
                    "missing_time_ms": round(track.missing_time_s * 1000.0, 3),
                    "motion_confirmed": track.motion_confirmed,
                    "observed_this_frame": track.observed_this_frame,
                }
                for track in self._tracks
            ],
        }
        return outputs

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_diagnostics = {}
        self._physical_rejections = 0
        self._preferred_track_id = None
        self._impact_recoveries = 0
        self._bounce_recoveries = 0
        self._primary_continuity_recoveries = 0
        self._direction_gate_rejections = 0
        self._nis_gate_rejections = 0
        self._acceleration_gate_rejections = 0
        self._contact_gate_rejections = 0
        self._contact_zones = []
        self._last_timestamp_s = None
        self._tracker_time_s = 0.0
        self._current_dt_s = self.default_dt_s

    def _resolve_dt(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None or not np.isfinite(timestamp_s):
            self._last_timestamp_s = None
            return self.default_dt_s
        timestamp = float(timestamp_s)
        if self._last_timestamp_s is None:
            dt_s = self.default_dt_s
        else:
            dt_s = timestamp - self._last_timestamp_s
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = self.default_dt_s
        self._last_timestamp_s = timestamp
        return dt_s

    def _shift_track_for_camera(self, track: _TrackState, dx: float, dy: float) -> None:
        if dx == 0.0 and dy == 0.0:
            return
        track.kalman.shift_position(dx, dy)
        track.last_observed_center = (
            track.last_observed_center[0] + dx,
            track.last_observed_center[1] + dy,
        )
        track.stationary_anchor_center = (
            track.stationary_anchor_center[0] + dx,
            track.stationary_anchor_center[1] + dy,
        )

    def _frame_scale(self, frame_width: int) -> float:
        if self.frame_scale_override is not None:
            return self.frame_scale_override
        return max(0.25, float(frame_width) / self.reference_frame_width)

    def _prediction_is_reliable(
        self,
        track: _TrackState,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if track.observed_this_frame:
            return True
        if track.confidence < self.prediction_min_confidence:
            return False
        scale = self._frame_scale(frame_width)
        speed = float(np.hypot(*track.kalman.velocity)) / scale
        limit_s = self._prediction_limit_s(speed)
        if track.missing_time_s > limit_s:
            return False
        x, y = track.center
        return 0.0 <= x <= float(frame_width) and 0.0 <= y <= float(frame_height)

    def _prediction_limit_s(self, normalized_speed: float) -> float:
        if self.fast_prediction_speed_px_per_second <= 0.0:
            return self.max_prediction_s
        if not self.continuous_prediction_horizon:
            return (
                min(self.max_prediction_s, self.fast_max_prediction_s)
                if normalized_speed >= self.fast_prediction_speed_px_per_second
                else self.max_prediction_s
            )
        speed_ratio = min(
            1.0,
            max(
                0.0,
                normalized_speed / self.fast_prediction_speed_px_per_second,
            ),
        )
        fast_limit = min(self.max_prediction_s, self.fast_max_prediction_s)
        return self.max_prediction_s - speed_ratio * (
            self.max_prediction_s - fast_limit
        )

    def _select_output_tracks(
        self,
        tracks: list[_TrackState],
        frame_width: int,
    ) -> list[_TrackState]:
        if not tracks:
            self._preferred_track_id = None
            return []
        if self.max_output_tracks == 0 or len(tracks) <= self.max_output_tracks:
            selected = list(tracks)
        else:
            scale = self._frame_scale(frame_width)

            def score(track: _TrackState) -> float:
                speed = float(np.hypot(*track.kalman.velocity)) / scale
                return (
                    (3.0 if track.observed_this_frame else 0.0)
                    + 2.0 * track.confidence
                    + 0.25 * min(speed / 300.0, 3.0)
                    + 0.50 * (track.track_id == self._preferred_track_id)
                    + 0.02 * min(track.hits, 10)
                    - 2.5 * track.missing_time_s
                    - 0.12 * track.stationary_frames
                )

            if self.observation_first_output:
                selected = sorted(
                    tracks,
                    key=lambda track: (
                        track.observed_this_frame,
                        score(track),
                    ),
                    reverse=True,
                )[: self.max_output_tracks]
            else:
                selected = sorted(
                    tracks,
                    key=score,
                    reverse=True,
                )[: self.max_output_tracks]
        if self.max_output_tracks == 1:
            self._preferred_track_id = selected[0].track_id
        return selected

    def _associate(
        self,
        candidate_track_indices: set[int],
        detections: list[BallDetection],
        frame_width: int,
    ) -> tuple[list[tuple[int, int, Optional[str]]], list[int]]:
        pairs: list[tuple[float, int, int, Optional[str]]] = []
        scale = self._frame_scale(frame_width)
        for track_index in candidate_track_indices:
            track = self._tracks[track_index]
            tx, ty = track.center
            vx, vy = track.kalman.velocity
            speed = float(np.hypot(vx, vy))
            elapsed_s = max(
                self.default_dt_s,
                self._tracker_time_s - track.last_observed_time_s,
            )
            dynamic_gate = min(
                self.max_gate_px * scale,
                self.base_gate_px * scale
                + speed * self._current_dt_s * self.velocity_gate_scale
                + track.missing_time_s * self.gate_growth_px_per_second * scale,
            )
            gate = max(dynamic_gate, self.tentative_gate_px * scale) if track.hits < self.min_hits else dynamic_gate
            for detection_index, detection in enumerate(detections):
                distance = float(np.hypot(detection.center[0] - tx, detection.center[1] - ty))
                observed_step = float(
                    np.hypot(
                        detection.center[0] - track.last_observed_center[0],
                        detection.center[1] - track.last_observed_center[1],
                    )
                )
                observed_speed = observed_step / elapsed_s
                observed_velocity = (
                    (float(detection.center[0]) - track.last_observed_center[0])
                    / elapsed_s,
                    (float(detection.center[1]) - track.last_observed_center[1])
                    / elapsed_s,
                )
                if (
                    self.max_speed_px_per_second > 0
                    and observed_speed > self.max_speed_px_per_second * scale
                ):
                    self._physical_rejections += 1
                    continue
                observed_acceleration = float(
                    np.hypot(
                        observed_velocity[0] - track.last_observed_velocity[0],
                        observed_velocity[1] - track.last_observed_velocity[1],
                    )
                ) / elapsed_s
                acceleration_violation = (
                    self.max_observed_acceleration_px_per_second2 > 0.0
                    and track.hits >= self.acceleration_gate_min_hits
                    and observed_acceleration
                    > self.max_observed_acceleration_px_per_second2 * scale
                )
                nis = (
                    track.kalman.innovation_nis(detection.center)
                    if self.use_nis_gate
                    else 0.0
                )
                nis_violation = (
                    self.use_nis_gate
                    and self.nis_gate_threshold > 0.0
                    and track.hits >= self.nis_gate_min_hits
                    and nis > self.nis_gate_threshold
                )
                direction_change_deg = self._direction_change_deg(
                    (vx, vy),
                    (
                        detection.center[0] - track.last_observed_center[0],
                        detection.center[1] - track.last_observed_center[1],
                    ),
                )
                direction_gate_active = (
                    self.max_flight_direction_change_deg > 0.0
                    and track.hits >= self.direction_gate_min_hits
                    and speed
                    >= self.direction_gate_min_speed_px_per_second * scale
                    and observed_speed
                    >= self.direction_gate_min_speed_px_per_second * scale
                )
                direction_violation = (
                    direction_gate_active
                    and direction_change_deg
                    > self.max_flight_direction_change_deg
                )
                impact_gate = min(
                    self.impact_recovery_gate_px * 1.5,
                    self.impact_recovery_gate_px
                    + elapsed_s * self.gate_growth_px_per_second,
                ) * scale
                impact_candidate = (
                    (
                        distance > gate
                        or direction_violation
                        or acceleration_violation
                        or nis_violation
                    )
                    and track.motion_confirmed
                    and detection.confidence >= self.high_conf
                    and max(speed, observed_speed)
                    >= self.impact_recovery_min_speed_px_per_second * scale
                    and observed_step <= impact_gate
                    and track.missing_time_s
                    >= self.impact_recovery_min_missing_s
                    and (
                        self.impact_recovery_max_missing_s is None
                        or track.missing_time_s
                        <= self.impact_recovery_max_missing_s
                    )
                )
                has_contact = self._has_contact_evidence(track, detection)
                contact_allowed = (
                    not self.require_contact_for_impact_recovery
                    or has_contact
                )
                impact_recovery = impact_candidate and contact_allowed
                bounce_recovery = self._is_bounce_recovery(
                    track,
                    detection,
                    observed_velocity,
                    scale,
                    has_contact=has_contact,
                )
                primary_continuity_recovery = (
                    (
                        direction_violation
                        or acceleration_violation
                        or nis_violation
                    )
                    and self._is_primary_continuity_recovery(
                        track,
                        detection,
                        observed_velocity,
                        observed_step,
                        scale,
                    )
                )
                if (
                    impact_candidate
                    and not contact_allowed
                    and not bounce_recovery
                    and not primary_continuity_recovery
                ):
                    self._contact_gate_rejections += 1
                recovery_kind = (
                    "bounce"
                    if bounce_recovery
                    else "primary_continuity"
                    if primary_continuity_recovery
                    else "impact"
                    if impact_recovery
                    else None
                )
                if recovery_kind is None and (
                    direction_violation
                    or acceleration_violation
                    or nis_violation
                ):
                    self._direction_gate_rejections += int(direction_violation)
                    self._acceleration_gate_rejections += int(
                        acceleration_violation
                    )
                    self._nis_gate_rejections += int(nis_violation)
                    self._physical_rejections += 1
                    continue
                if distance > gate and recovery_kind is None:
                    continue
                dw = max(1.0, detection.bbox[2] - detection.bbox[0])
                dh = max(1.0, detection.bbox[3] - detection.bbox[1])
                size_penalty = abs(np.log(dw / max(track.bbox_width, 1.0))) + abs(
                    np.log(dh / max(track.bbox_height, 1.0))
                )
                if bounce_recovery or primary_continuity_recovery:
                    # A strict rebound signature on an established track should
                    # stay ahead of a one-frame tentative clone. The same
                    # priority applies to consecutive primary observations that
                    # only disagree with a lagging filter state.
                    cost = -0.25 + 0.25 * observed_step / max(impact_gate, 1.0)
                elif impact_recovery:
                    cost = 1.0 + observed_step / max(impact_gate, 1.0)
                else:
                    cost = distance / max(gate, 1.0)
                cost += 0.08 * size_penalty - 0.10 * detection.confidence
                if self.use_nis_gate and self.nis_gate_threshold > 0.0:
                    cost += 0.03 * min(nis / self.nis_gate_threshold, 3.0)
                pairs.append((cost, track_index, detection_index, recovery_kind))

        matches: list[tuple[int, int, Optional[str]]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index, recovery_kind in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index, recovery_kind))
            if recovery_kind == "impact":
                self._impact_recoveries += 1
            elif recovery_kind == "bounce":
                self._bounce_recoveries += 1
            elif recovery_kind == "primary_continuity":
                self._primary_continuity_recoveries += 1
        unmatched_detections = [
            index for index in range(len(detections)) if index not in used_detections
        ]
        return matches, unmatched_detections

    def _is_primary_continuity_recovery(
        self,
        track: _TrackState,
        detection: BallDetection,
        observed_velocity: tuple[float, float],
        observed_step: float,
        scale: float,
    ) -> bool:
        """Let consecutive primary observations correct a lagging filter."""
        if (
            not self.primary_continuity_recovery_enabled
            or not track.motion_confirmed
            or track.missing_time_s > 1e-9
            or detection.confidence < self.high_conf
        ):
            return False
        detection_source = str(detection.source).strip().lower()
        previous_source = str(track.last_observation_source).strip().lower()
        if (
            not self.primary_observation_sources
            or detection_source not in self.primary_observation_sources
            or previous_source not in self.primary_observation_sources
        ):
            return False
        if observed_step > self.primary_continuity_gate_px * scale:
            return False
        previous_speed = float(np.hypot(*track.last_observed_velocity))
        observed_speed = float(np.hypot(*observed_velocity))
        direction_floor = self.direction_gate_min_speed_px_per_second * scale
        if previous_speed < direction_floor or observed_speed < direction_floor:
            return True
        raw_direction_change = self._direction_change_deg(
            track.last_observed_velocity,
            observed_velocity,
        )
        return raw_direction_change <= self.max_flight_direction_change_deg

    def _is_bounce_recovery(
        self,
        track: _TrackState,
        detection: BallDetection,
        observed_velocity: tuple[float, float],
        scale: float,
        *,
        has_contact: bool,
    ) -> bool:
        """Recognize a short, model-observed image-plane rebound."""
        if not self.bounce_recovery_enabled or not track.motion_confirmed:
            return False
        if detection.confidence < self.high_conf:
            return False
        if (
            self.bounce_recovery_sources
            and str(detection.source).strip().lower()
            not in self.bounce_recovery_sources
        ):
            return False
        if self.bounce_recovery_require_no_contact and has_contact:
            return False
        if track.missing_time_s > self.bounce_recovery_max_missing_s:
            return False
        displacement = float(
            np.hypot(
                float(detection.center[0]) - track.last_observed_center[0],
                float(detection.center[1]) - track.last_observed_center[1],
            )
        )
        if displacement > self.bounce_recovery_max_displacement_px * scale:
            return False
        previous_vx, previous_vy = track.last_observed_velocity
        observed_vx, observed_vy = observed_velocity
        rebound_start = (
            previous_vy
            >= self.bounce_recovery_min_downward_speed_px_per_second * scale
            and observed_vy
            <= -self.bounce_recovery_min_upward_speed_px_per_second * scale
        )
        rebound_stabilizing = (
            self._tracker_time_s <= track.bounce_recovery_until_s
            and previous_vy
            <= -self.bounce_recovery_min_upward_speed_px_per_second * scale
            and observed_vy
            <= -self.bounce_recovery_min_upward_speed_px_per_second * scale
        )
        if not rebound_start and not rebound_stabilizing:
            return False
        horizontal_floor = (
            self.bounce_recovery_min_horizontal_speed_px_per_second * scale
        )
        if (
            abs(previous_vx) >= horizontal_floor
            and abs(observed_vx) >= horizontal_floor
            and previous_vx * observed_vx < 0.0
        ):
            return False
        return float(detection.center[1]) <= track.last_observed_center[1]

    def _has_contact_evidence(
        self,
        track: _TrackState,
        detection: BallDetection,
    ) -> bool:
        """Check whether an observed or predicted ball point is near a player."""
        start = track.last_observed_center
        end = (float(detection.center[0]), float(detection.center[1]))
        predicted = track.center
        for zone in self._contact_zones:
            if len(zone) != 4:
                continue
            person_height = max(1.0, float(zone[3]) - float(zone[1]))
            margin = person_height * self.contact_margin_ratio
            expanded = [
                float(zone[0]) - margin,
                float(zone[1]) - margin,
                float(zone[2]) + margin,
                float(zone[3]) + margin,
            ]
            if (
                self._point_in_rect(start, expanded)
                or self._point_in_rect(end, expanded)
                or self._point_in_rect(predicted, expanded)
            ):
                return True
        return False

    @staticmethod
    def _point_in_rect(point: tuple[float, float], rect: list[float]) -> bool:
        return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]

    @staticmethod
    def _direction_change_deg(
        previous_velocity: tuple[float, float],
        observed_displacement: tuple[float, float],
    ) -> float:
        previous = np.asarray(previous_velocity, dtype=np.float64)
        observed = np.asarray(observed_displacement, dtype=np.float64)
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(observed))
        if denominator <= 1e-9:
            return 0.0
        cosine = float(np.clip(np.dot(previous, observed) / denominator, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _apply_matches(
        self,
        matches: list[tuple[int, int, Optional[str]]],
        detections: list[BallDetection],
        frame_width: int,
    ) -> None:
        scale = self._frame_scale(frame_width)
        for track_index, detection_index, recovery_kind in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            elapsed_s = max(
                self.default_dt_s,
                self._tracker_time_s - track.last_observed_time_s,
            )
            measured_velocity = (
                (float(detection.center[0]) - track.last_observed_center[0]) / elapsed_s,
                (float(detection.center[1]) - track.last_observed_center[1]) / elapsed_s,
            )
            anchor_displacement = float(
                np.hypot(
                    detection.center[0] - track.stationary_anchor_center[0],
                    detection.center[1] - track.stationary_anchor_center[1],
                )
            )
            raw_anchor_displacement = float(
                np.hypot(
                    detection.center[0] - track.raw_stationary_anchor_center[0],
                    detection.center[1] - track.raw_stationary_anchor_center[1],
                )
            )
            track.stationary_anchor_frames += max(1, track.missing_frames + 1)
            track.raw_stationary_anchor_frames += max(1, track.missing_frames + 1)
            anchor_elapsed_s = max(
                self.default_dt_s,
                self._tracker_time_s - track.stationary_anchor_time_s,
            )
            raw_anchor_elapsed_s = max(
                self.default_dt_s,
                self._tracker_time_s - track.raw_stationary_anchor_time_s,
            )
            average_anchor_speed = anchor_displacement / anchor_elapsed_s
            raw_average_anchor_speed = raw_anchor_displacement / max(
                raw_anchor_elapsed_s,
                self.default_dt_s,
            )
            if (
                anchor_displacement > self.motion_threshold_px * scale
                and average_anchor_speed
                >= self.min_motion_speed_px_per_second * scale
                and raw_anchor_displacement > self.motion_threshold_px * scale
                and raw_average_anchor_speed
                >= self.min_motion_speed_px_per_second * scale
            ):
                track.motion_confirmed = True
                track.stationary_frames = 0
                track.stationary_time_s = 0.0
                track.stationary_anchor_center = (
                    float(detection.center[0]),
                    float(detection.center[1]),
                )
                track.stationary_anchor_frames = 0
                track.stationary_anchor_time_s = self._tracker_time_s
                track.raw_stationary_anchor_center = (
                    float(detection.center[0]),
                    float(detection.center[1]),
                )
                track.raw_stationary_anchor_frames = 0
                track.raw_stationary_anchor_time_s = self._tracker_time_s
            else:
                track.stationary_frames += 1
                track.stationary_time_s += elapsed_s
            track.last_observed_center = (float(detection.center[0]), float(detection.center[1]))
            track.raw_last_observed_center = (
                float(detection.center[0]),
                float(detection.center[1]),
            )
            track.kalman.update(detection.center)
            if recovery_kind is not None:
                track.kalman.set_velocity(*measured_velocity)
                track.kalman.reset_acceleration()
                if (
                    recovery_kind == "bounce"
                    and self._tracker_time_s > track.bounce_recovery_until_s
                ):
                    track.bounce_recovery_until_s = (
                        self._tracker_time_s + self.bounce_recovery_max_missing_s
                    )
            elif track.hits + 1 < self.constant_acceleration_min_observations:
                track.kalman.reset_acceleration()
            elif self.max_acceleration_px_per_second2 > 0.0:
                track.kalman.clamp_acceleration(
                    self.max_acceleration_px_per_second2 * scale
                )
            track.last_observed_velocity = measured_velocity
            measured_width = max(1.0, detection.bbox[2] - detection.bbox[0])
            measured_height = max(1.0, detection.bbox[3] - detection.bbox[1])
            track.bbox_width = 0.7 * measured_width + 0.3 * track.bbox_width
            track.bbox_height = 0.7 * measured_height + 0.3 * track.bbox_height
            track.confidence = detection.confidence
            track.last_observation_source = detection.source
            track.hits += 1
            track.missing_frames = 0
            track.missing_time_s = 0.0
            track.last_observed_time_s = self._tracker_time_s
            track.observed_this_frame = True

    def _create_track(self, detection: BallDetection) -> None:
        if self.motion_model == "constant_acceleration":
            kalman = ConstantAccelerationKalman(
                self.acceleration_process_noise,
                self.measurement_noise,
            )
        else:
            kalman = ConstantVelocityKalman(
                self.process_noise,
                self.measurement_noise,
            )
        kalman.initialize(detection.center)
        self._tracks.append(
            _TrackState(
                track_id=self._next_track_id,
                kalman=kalman,
                bbox_width=max(1.0, detection.bbox[2] - detection.bbox[0]),
                bbox_height=max(1.0, detection.bbox[3] - detection.bbox[1]),
                confidence=detection.confidence,
                last_observed_center=(float(detection.center[0]), float(detection.center[1])),
                stationary_anchor_center=(float(detection.center[0]), float(detection.center[1])),
                raw_stationary_anchor_center=(float(detection.center[0]), float(detection.center[1])),
                raw_last_observed_center=(float(detection.center[0]), float(detection.center[1])),
                motion_confirmed=not self.require_motion_confirmation,
                last_observed_time_s=self._tracker_time_s,
                stationary_anchor_time_s=self._tracker_time_s,
                raw_stationary_anchor_time_s=self._tracker_time_s,
                last_observation_source=detection.source,
            )
        )
        self._next_track_id += 1

    def _to_output(self, track: _TrackState, frame_width: int, frame_height: int) -> BallTrack:
        cx, cy = track.center
        bbox = track.bbox
        bbox = [
            min(max(bbox[0], 0.0), float(frame_width)),
            min(max(bbox[1], 0.0), float(frame_height)),
            min(max(bbox[2], 0.0), float(frame_width)),
            min(max(bbox[3], 0.0), float(frame_height)),
        ]
        status = "observed" if track.observed_this_frame else "predicted"
        return BallTrack(
            track_id=track.track_id,
            status=status,
            center=[cx, cy],
            bbox=bbox,
            confidence=track.confidence,
            velocity=list(track.kalman.velocity),
            acceleration=list(track.kalman.acceleration),
            motion_model=self.motion_model,
            missing_frames=track.missing_frames,
            missing_time_ms=track.missing_time_s * 1000.0,
            source="detector" if track.observed_this_frame else "prediction",
            observation_source=(
                track.last_observation_source
                if track.observed_this_frame
                else None
            ),
            age=track.age,
            hits=track.hits,
            confirmed=track.hits >= self.min_hits,
            motion_confirmed=track.motion_confirmed,
            stationary_frames=track.stationary_frames,
        )
