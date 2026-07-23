"""Lightweight multi-pickleball tracking with short-term motion prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import BallDetection, BallTrack


class ConstantVelocityKalman:
    """Small NumPy Kalman filter with state ``x, y, vx, vy``."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.p = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = False
        self.h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    def initialize(self, center: list[float]) -> None:
        self.x[:, 0] = (float(center[0]), float(center[1]), 0.0, 0.0)
        self.p = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = True

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        q = self._process_noise**2
        q_matrix = np.array(
            [
                [q * dt**4 / 4, 0.0, q * dt**3 / 2, 0.0],
                [0.0, q * dt**4 / 4, 0.0, q * dt**3 / 2],
                [q * dt**3 / 2, 0.0, q * dt**2, 0.0],
                [0.0, q * dt**3 / 2, 0.0, q * dt**2],
            ],
            dtype=np.float64,
        )
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q_matrix
        return self.position

    def update(self, center: list[float]) -> None:
        if not self.initialized:
            self.initialize(center)
            return
        measurement = np.array([[float(center[0])], [float(center[1])]])
        residual = measurement - self.h @ self.x
        innovation = self.h @ self.p @ self.h.T + np.eye(2) * self._measurement_noise**2
        gain = self.p @ self.h.T @ np.linalg.inv(innovation)
        self.x = self.x + gain @ residual
        self.p = (np.eye(4) - gain @ self.h) @ self.p

    def shift_position(self, dx: float, dy: float) -> None:
        self.x[0, 0] += float(dx)
        self.x[1, 0] += float(dy)

    def set_velocity(self, vx: float, vy: float) -> None:
        self.x[2, 0] = float(vx)
        self.x[3, 0] = float(vy)

    def damp_velocity(self, factor: float) -> None:
        self.x[2:, 0] *= float(factor)

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


@dataclass
class _TrackState:
    track_id: int
    kalman: ConstantVelocityKalman
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
    trajectory until ``max_predict_frames`` is reached.
    """

    def __init__(
        self,
        high_conf: float = 0.25,
        low_conf: float = 0.08,
        min_hits: int = 2,
        max_predict_frames: int = 8,
        max_missing_frames: int = 20,
        base_gate_px: float = 70.0,
        max_gate_px: float = 280.0,
        gate_growth_px: float = 20.0,
        velocity_gate_scale: float = 1.5,
        tentative_gate_px: float = 150.0,
        max_speed_px_per_frame: float = 0.0,
        reference_frame_width: float = 1280.0,
        require_motion_confirmation: bool = False,
        motion_threshold_px: float = 5.0,
        max_stationary_frames: int = 0,
        unconfirmed_max_frames: int = 30,
        max_output_tracks: int = 0,
        impact_recovery_gate_px: float = 260.0,
        impact_recovery_min_speed_px: float = 15.0,
        prediction_min_confidence: float = 0.0,
        fast_prediction_speed_px: float = 0.0,
        fast_max_predict_frames: int = 3,
        prediction_velocity_decay: float = 1.0,
        min_motion_speed_px: float = 0.0,
        confidence_decay: float = 0.82,
        process_noise: float = 12.0,
        measurement_noise: float = 4.0,
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
        self.max_predict_frames = int(max_predict_frames)
        self.max_missing_frames = int(max_missing_frames)
        self.base_gate_px = float(base_gate_px)
        self.max_gate_px = float(max_gate_px)
        self.gate_growth_px = float(gate_growth_px)
        self.velocity_gate_scale = float(velocity_gate_scale)
        self.tentative_gate_px = float(tentative_gate_px)
        self.max_speed_px_per_frame = float(max_speed_px_per_frame)
        self.reference_frame_width = max(1.0, float(reference_frame_width))
        self.require_motion_confirmation = bool(require_motion_confirmation)
        self.motion_threshold_px = max(0.0, float(motion_threshold_px))
        self.max_stationary_frames = max(0, int(max_stationary_frames))
        self.unconfirmed_max_frames = max(1, int(unconfirmed_max_frames))
        self.max_output_tracks = max(0, int(max_output_tracks))
        self.impact_recovery_gate_px = max(0.0, float(impact_recovery_gate_px))
        self.impact_recovery_min_speed_px = max(0.0, float(impact_recovery_min_speed_px))
        self.prediction_min_confidence = max(0.0, float(prediction_min_confidence))
        self.fast_prediction_speed_px = max(0.0, float(fast_prediction_speed_px))
        self.fast_max_predict_frames = max(0, int(fast_max_predict_frames))
        self.prediction_velocity_decay = min(1.0, max(0.0, float(prediction_velocity_decay)))
        self.min_motion_speed_px = max(0.0, float(min_motion_speed_px))
        self.confidence_decay = float(confidence_decay)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_tracks = max(1, int(max_tracks))
        self.emit_tentative = bool(emit_tentative)
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1
        self._last_diagnostics: dict = {}
        self._physical_rejections = 0
        self._preferred_track_id: Optional[int] = None
        self._impact_recoveries = 0

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
    ) -> list[BallTrack]:
        valid = [d for d in detections if d.confidence >= self.low_conf]
        high = [d for d in valid if d.confidence >= self.high_conf]
        low = [d for d in valid if d.confidence < self.high_conf]

        camera_dx, camera_dy = (float(camera_motion[0]), float(camera_motion[1]))
        for track in self._tracks:
            self._shift_track_for_camera(track, camera_dx, camera_dy)
            track.kalman.predict()
            track.age += 1
            track.observed_this_frame = False

        unmatched_tracks = set(range(len(self._tracks)))
        self._physical_rejections = 0
        self._impact_recoveries = 0
        high_matches, unmatched_high = self._associate(unmatched_tracks, high, frame_width)
        self._apply_matches(high_matches, high, frame_width)
        unmatched_tracks -= {track_index for track_index, _, _ in high_matches}

        low_matches, _ = self._associate(unmatched_tracks, low, frame_width)
        self._apply_matches(low_matches, low, frame_width)
        unmatched_tracks -= {track_index for track_index, _, _ in low_matches}

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.missing_frames += 1
            track.confidence *= self.confidence_decay
            track.kalman.damp_velocity(self.prediction_velocity_decay)

        stale_missing = {
            index for index, track in enumerate(self._tracks)
            if track.missing_frames > self.max_missing_frames
        }
        stale_unconfirmed = {
            index for index, track in enumerate(self._tracks)
            if self.require_motion_confirmation
            and not track.motion_confirmed
            and track.age > self.unconfirmed_max_frames
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
            if track.missing_frames <= self.max_predict_frames
            and self._prediction_is_reliable(track, frame_width, frame_height)
            and (self.emit_tentative or track.hits >= self.min_hits)
            and (not self.require_motion_confirmation or track.motion_confirmed)
            and (
                self.max_stationary_frames == 0
                or track.stationary_frames <= self.max_stationary_frames
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
            "camera_motion_dx": camera_dx,
            "camera_motion_dy": camera_dy,
            "motion_unconfirmed_tracks": sum(not track.motion_confirmed for track in self._tracks),
            "stationary_suppressed_tracks": sum(
                self.max_stationary_frames > 0
                and track.stationary_frames > self.max_stationary_frames
                for track in self._tracks
            ),
            "track_states": [
                {
                    "track_id": track.track_id,
                    "center": [round(value, 3) for value in track.center],
                    "velocity": [round(value, 3) for value in track.kalman.velocity],
                    "missing_frames": track.missing_frames,
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
        scale = max(0.25, float(frame_width) / self.reference_frame_width)
        speed = float(np.hypot(*track.kalman.velocity)) / scale
        limit = self.max_predict_frames
        if self.fast_prediction_speed_px > 0 and speed >= self.fast_prediction_speed_px:
            limit = min(limit, self.fast_max_predict_frames)
        if track.missing_frames > limit:
            return False
        x, y = track.center
        return 0.0 <= x <= float(frame_width) and 0.0 <= y <= float(frame_height)

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
            scale = max(0.25, float(frame_width) / self.reference_frame_width)

            def score(track: _TrackState) -> float:
                speed = float(np.hypot(*track.kalman.velocity)) / scale
                return (
                    (3.0 if track.observed_this_frame else 0.0)
                    + 2.0 * track.confidence
                    + 0.25 * min(speed / 10.0, 3.0)
                    + 0.50 * (track.track_id == self._preferred_track_id)
                    + 0.02 * min(track.hits, 10)
                    - 0.25 * track.missing_frames
                    - 0.12 * track.stationary_frames
                )

            selected = sorted(tracks, key=score, reverse=True)[: self.max_output_tracks]
        if self.max_output_tracks == 1:
            self._preferred_track_id = selected[0].track_id
        return selected

    def _associate(
        self,
        candidate_track_indices: set[int],
        detections: list[BallDetection],
        frame_width: int,
    ) -> tuple[list[tuple[int, int, bool]], list[int]]:
        pairs: list[tuple[float, int, int, bool]] = []
        scale = max(0.25, float(frame_width) / self.reference_frame_width)
        for track_index in candidate_track_indices:
            track = self._tracks[track_index]
            tx, ty = track.center
            vx, vy = track.kalman.velocity
            speed = float(np.hypot(vx, vy))
            dynamic_gate = min(
                self.max_gate_px * scale,
                self.base_gate_px * scale
                + speed * self.velocity_gate_scale
                + track.missing_frames * self.gate_growth_px * scale,
            )
            gate = max(dynamic_gate, self.tentative_gate_px * scale) if track.hits < self.min_hits else dynamic_gate
            for detection_index, detection in enumerate(detections):
                distance = float(np.hypot(detection.center[0] - tx, detection.center[1] - ty))
                elapsed_frames = max(1, track.missing_frames + 1)
                observed_step = float(
                    np.hypot(
                        detection.center[0] - track.last_observed_center[0],
                        detection.center[1] - track.last_observed_center[1],
                    )
                )
                observed_speed = observed_step / elapsed_frames
                if (
                    self.max_speed_px_per_frame > 0
                    and observed_step > self.max_speed_px_per_frame * scale * elapsed_frames
                ):
                    self._physical_rejections += 1
                    continue
                impact_gate = min(
                    self.impact_recovery_gate_px * 1.5,
                    self.impact_recovery_gate_px
                    + elapsed_frames * self.gate_growth_px,
                ) * scale
                impact_recovery = (
                    distance > gate
                    and track.motion_confirmed
                    and detection.confidence >= self.high_conf
                    and max(speed, observed_speed) >= self.impact_recovery_min_speed_px * scale
                    and observed_step <= impact_gate
                )
                if distance > gate and not impact_recovery:
                    continue
                dw = max(1.0, detection.bbox[2] - detection.bbox[0])
                dh = max(1.0, detection.bbox[3] - detection.bbox[1])
                size_penalty = abs(np.log(dw / max(track.bbox_width, 1.0))) + abs(
                    np.log(dh / max(track.bbox_height, 1.0))
                )
                if impact_recovery:
                    cost = 1.0 + observed_step / max(impact_gate, 1.0)
                else:
                    cost = distance / max(gate, 1.0)
                cost += 0.08 * size_penalty - 0.10 * detection.confidence
                pairs.append((cost, track_index, detection_index, impact_recovery))

        matches: list[tuple[int, int, bool]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index, impact_recovery in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index, impact_recovery))
            if impact_recovery:
                self._impact_recoveries += 1
        unmatched_detections = [
            index for index in range(len(detections)) if index not in used_detections
        ]
        return matches, unmatched_detections

    def _apply_matches(
        self,
        matches: list[tuple[int, int, bool]],
        detections: list[BallDetection],
        frame_width: int,
    ) -> None:
        scale = max(0.25, float(frame_width) / self.reference_frame_width)
        for track_index, detection_index, impact_recovery in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            elapsed_frames = max(1, track.missing_frames + 1)
            measured_velocity = (
                (float(detection.center[0]) - track.last_observed_center[0]) / elapsed_frames,
                (float(detection.center[1]) - track.last_observed_center[1]) / elapsed_frames,
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
            track.stationary_anchor_frames += elapsed_frames
            track.raw_stationary_anchor_frames += elapsed_frames
            average_anchor_speed = anchor_displacement / max(track.stationary_anchor_frames, 1)
            raw_average_anchor_speed = raw_anchor_displacement / max(
                track.raw_stationary_anchor_frames,
                1,
            )
            if (
                anchor_displacement > self.motion_threshold_px * scale
                and average_anchor_speed >= self.min_motion_speed_px * scale
                and raw_anchor_displacement > self.motion_threshold_px * scale
                and raw_average_anchor_speed >= self.min_motion_speed_px * scale
            ):
                track.motion_confirmed = True
                track.stationary_frames = 0
                track.stationary_anchor_center = (
                    float(detection.center[0]),
                    float(detection.center[1]),
                )
                track.stationary_anchor_frames = 0
                track.raw_stationary_anchor_center = (
                    float(detection.center[0]),
                    float(detection.center[1]),
                )
                track.raw_stationary_anchor_frames = 0
            else:
                track.stationary_frames += 1
            track.last_observed_center = (float(detection.center[0]), float(detection.center[1]))
            track.raw_last_observed_center = (
                float(detection.center[0]),
                float(detection.center[1]),
            )
            track.kalman.update(detection.center)
            if impact_recovery:
                track.kalman.set_velocity(*measured_velocity)
            measured_width = max(1.0, detection.bbox[2] - detection.bbox[0])
            measured_height = max(1.0, detection.bbox[3] - detection.bbox[1])
            track.bbox_width = 0.7 * measured_width + 0.3 * track.bbox_width
            track.bbox_height = 0.7 * measured_height + 0.3 * track.bbox_height
            track.confidence = detection.confidence
            track.hits += 1
            track.missing_frames = 0
            track.observed_this_frame = True

    def _create_track(self, detection: BallDetection) -> None:
        kalman = ConstantVelocityKalman(self.process_noise, self.measurement_noise)
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
            missing_frames=track.missing_frames,
            source="detector" if track.observed_this_frame else "prediction",
            age=track.age,
            hits=track.hits,
            confirmed=track.hits >= self.min_hits,
            motion_confirmed=track.motion_confirmed,
            stationary_frames=track.stationary_frames,
        )
