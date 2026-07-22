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
        self.confidence_decay = float(confidence_decay)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_tracks = max(1, int(max_tracks))
        self.emit_tentative = bool(emit_tentative)
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1
        self._last_diagnostics: dict = {}

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
    ) -> list[BallTrack]:
        valid = [d for d in detections if d.confidence >= self.low_conf]
        high = [d for d in valid if d.confidence >= self.high_conf]
        low = [d for d in valid if d.confidence < self.high_conf]

        for track in self._tracks:
            track.kalman.predict()
            track.age += 1
            track.observed_this_frame = False

        unmatched_tracks = set(range(len(self._tracks)))
        high_matches, unmatched_high = self._associate(unmatched_tracks, high)
        self._apply_matches(high_matches, high)
        unmatched_tracks -= {track_index for track_index, _ in high_matches}

        low_matches, _ = self._associate(unmatched_tracks, low)
        self._apply_matches(low_matches, low)
        unmatched_tracks -= {track_index for track_index, _ in low_matches}

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.missing_frames += 1
            track.confidence *= self.confidence_decay

        created = 0
        for detection_index in unmatched_high:
            if len(self._tracks) >= self.max_tracks:
                break
            self._create_track(high[detection_index])
            created += 1

        removed = sum(track.missing_frames > self.max_missing_frames for track in self._tracks)
        self._tracks = [
            track for track in self._tracks if track.missing_frames <= self.max_missing_frames
        ]

        outputs = [
            self._to_output(track, frame_width, frame_height)
            for track in self._tracks
            if track.missing_frames <= self.max_predict_frames
            and (self.emit_tentative or track.hits >= self.min_hits)
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
        }
        return outputs

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_diagnostics = {}

    def _associate(
        self,
        candidate_track_indices: set[int],
        detections: list[BallDetection],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        pairs: list[tuple[float, int, int]] = []
        for track_index in candidate_track_indices:
            track = self._tracks[track_index]
            tx, ty = track.center
            vx, vy = track.kalman.velocity
            speed = float(np.hypot(vx, vy))
            gate = min(
                self.max_gate_px,
                self.base_gate_px
                + speed * self.velocity_gate_scale
                + track.missing_frames * self.gate_growth_px,
            )
            for detection_index, detection in enumerate(detections):
                distance = float(np.hypot(detection.center[0] - tx, detection.center[1] - ty))
                if distance > gate:
                    continue
                dw = max(1.0, detection.bbox[2] - detection.bbox[0])
                dh = max(1.0, detection.bbox[3] - detection.bbox[1])
                size_penalty = abs(np.log(dw / max(track.bbox_width, 1.0))) + abs(
                    np.log(dh / max(track.bbox_height, 1.0))
                )
                cost = distance / max(gate, 1.0) + 0.08 * size_penalty - 0.10 * detection.confidence
                pairs.append((cost, track_index, detection_index))

        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))
        unmatched_detections = [
            index for index in range(len(detections)) if index not in used_detections
        ]
        return matches, unmatched_detections

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        detections: list[BallDetection],
    ) -> None:
        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.kalman.update(detection.center)
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
        )
