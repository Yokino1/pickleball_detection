"""Lightweight person-box tracking and player-versus-spectator selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import PlayerDetection


def _bbox_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


@dataclass
class _PersonTrack:
    track_id: int
    bbox: list[float]
    confidence: float
    velocity: tuple[float, float] = (0.0, 0.0)
    last_detection_time_s: float = 0.0
    missed_detection_runs: int = 0
    hits: int = 1


class PersonBoxTracker:
    """Carry person boxes across frames between low-frequency detections."""

    def __init__(
        self,
        max_missing_detection_runs: int = 3,
        association_iou: float = 0.10,
        center_gate_height_scale: float = 0.80,
        velocity_decay: float = 0.80,
        default_fps: float = 30.0,
    ):
        self.max_missing_detection_runs = max(0, int(max_missing_detection_runs))
        self.association_iou = max(0.0, float(association_iou))
        self.center_gate_height_scale = max(0.1, float(center_gate_height_scale))
        self.velocity_decay = min(1.0, max(0.0, float(velocity_decay)))
        self.default_dt_s = 1.0 / max(1.0, float(default_fps))
        self._tracks: list[_PersonTrack] = []
        self._next_track_id = 1
        self._last_time_s: Optional[float] = None
        self._time_s = 0.0

    def update(
        self,
        detections: Optional[list[PlayerDetection]],
        frame_width: int,
        frame_height: int,
        timestamp_s: Optional[float] = None,
    ) -> list[PlayerDetection]:
        dt_s = self._resolve_dt(timestamp_s)
        self._time_s += dt_s
        for track in self._tracks:
            dx, dy = track.velocity[0] * dt_s, track.velocity[1] * dt_s
            track.bbox = [
                track.bbox[0] + dx,
                track.bbox[1] + dy,
                track.bbox[2] + dx,
                track.bbox[3] + dy,
            ]
            track.velocity = (
                track.velocity[0] * self.velocity_decay,
                track.velocity[1] * self.velocity_decay,
            )

        if detections is not None:
            self._associate_detections(detections)

        outputs = []
        for track in self._tracks:
            bbox = self._clip_bbox(track.bbox, frame_width, frame_height)
            outputs.append(
                PlayerDetection(
                    bbox=bbox,
                    track_id=track.track_id,
                    confidence=track.confidence,
                    foot_point=[(bbox[0] + bbox[2]) / 2.0, bbox[3]],
                    source="person_track",
                    track_hits=track.hits,
                )
            )
        return outputs

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_time_s = None
        self._time_s = 0.0

    def _associate_detections(self, detections: list[PlayerDetection]) -> None:
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            tx, ty = _center(track.bbox)
            track_height = max(1.0, track.bbox[3] - track.bbox[1])
            for detection_index, detection in enumerate(detections):
                dx, dy = _center(detection.bbox)
                distance = float(np.hypot(dx - tx, dy - ty))
                gate = max(
                    track_height,
                    detection.bbox[3] - detection.bbox[1],
                ) * self.center_gate_height_scale
                iou = _bbox_iou(track.bbox, detection.bbox)
                if iou >= self.association_iou or distance <= gate:
                    pairs.append((distance / max(gate, 1.0) - iou, track_index, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            track = self._tracks[track_index]
            detection = detections[detection_index]
            previous_center = _center(track.bbox)
            new_center = _center(detection.bbox)
            elapsed_s = max(
                self.default_dt_s,
                self._time_s - track.last_detection_time_s,
            )
            track.velocity = (
                (new_center[0] - previous_center[0]) / elapsed_s,
                (new_center[1] - previous_center[1]) / elapsed_s,
            )
            track.bbox = list(detection.bbox)
            track.confidence = float(detection.confidence)
            track.last_detection_time_s = self._time_s
            track.missed_detection_runs = 0
            track.hits += 1
            used_tracks.add(track_index)
            used_detections.add(detection_index)

        for index, track in enumerate(self._tracks):
            if index not in used_tracks:
                track.missed_detection_runs += 1
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_detection_runs <= self.max_missing_detection_runs
        ]
        for index, detection in enumerate(detections):
            if index in used_detections:
                continue
            self._tracks.append(
                _PersonTrack(
                    track_id=self._next_track_id,
                    bbox=list(detection.bbox),
                    confidence=float(detection.confidence),
                    last_detection_time_s=self._time_s,
                )
            )
            self._next_track_id += 1

    def _resolve_dt(self, timestamp_s: Optional[float]) -> float:
        if timestamp_s is None or not np.isfinite(timestamp_s):
            self._last_time_s = None
            return self.default_dt_s
        timestamp = float(timestamp_s)
        if self._last_time_s is None:
            dt_s = self.default_dt_s
        else:
            dt_s = timestamp - self._last_time_s
            if not np.isfinite(dt_s) or dt_s <= 0.0:
                dt_s = self.default_dt_s
        self._last_time_s = timestamp
        return dt_s

    @staticmethod
    def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
        return [
            min(max(0.0, bbox[0]), float(width)),
            min(max(0.0, bbox[1]), float(height)),
            min(max(0.0, bbox[2]), float(width)),
            min(max(0.0, bbox[3]), float(height)),
        ]


class PlayerSelector:
    """Select likely court players from all tracked person boxes."""

    def __init__(
        self,
        max_players: int = 4,
        play_area_normalized: Optional[list[float]] = None,
        spectator_exclusion_regions: Optional[list[list[float]]] = None,
        min_track_hits: int = 1,
    ):
        self.max_players = max(1, int(max_players))
        self.play_area_normalized = play_area_normalized or [0.0, 0.20, 1.0, 1.0]
        self.spectator_exclusion_regions = spectator_exclusion_regions or []
        self.min_track_hits = max(1, int(min_track_hits))

    def select(
        self,
        people: list[PlayerDetection],
        frame_width: int,
        frame_height: int,
    ) -> list[PlayerDetection]:
        ranked: list[tuple[float, PlayerDetection]] = []
        for person in people:
            foot = person.foot_point or [
                (person.bbox[0] + person.bbox[2]) / 2.0,
                person.bbox[3],
            ]
            normalized_foot = [
                foot[0] / max(1.0, float(frame_width)),
                foot[1] / max(1.0, float(frame_height)),
            ]
            in_play_area = self._point_in_rect(normalized_foot, self.play_area_normalized)
            excluded = any(
                self._point_in_rect(normalized_foot, region)
                for region in self.spectator_exclusion_regions
            )
            height_ratio = max(0.0, person.bbox[3] - person.bbox[1]) / max(
                1.0, float(frame_height)
            )
            score = (
                float(person.confidence)
                + 0.75 * normalized_foot[1]
                + 0.25 * min(height_ratio, 1.0)
                + 0.05 * min(person.track_hits, 10)
            )
            person.selection_score = score
            person.eligible_player = (
                in_play_area
                and not excluded
                and person.track_hits >= self.min_track_hits
            )
            if person.eligible_player:
                ranked.append((score, person))

        selected_ids = {
            person.track_id
            for _, person in sorted(ranked, key=lambda item: item[0], reverse=True)[
                : self.max_players
            ]
        }
        for person in people:
            person.eligible_player = person.track_id in selected_ids
        return people

    @staticmethod
    def _point_in_rect(point: list[float], rect: list[float]) -> bool:
        return (
            len(rect) == 4
            and rect[0] <= point[0] <= rect[2]
            and rect[1] <= point[1] <= rect[3]
        )
