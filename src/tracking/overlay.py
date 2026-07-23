"""OpenCV overlay for multi-ball tracks, predictions, and trails."""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from .types import BallTrack, FrameResult


_PALETTE = (
    (0, 255, 0),
    (0, 255, 255),
    (0, 165, 255),
    (0, 0, 255),
    (0, 210, 140),
    (0, 120, 255),
    (0, 255, 190),
)


class TrackingOverlay:
    def __init__(self, trail_length: int = 30, draw_raw_detections: bool = False):
        self.trail_length = max(2, int(trail_length))
        self.draw_raw_detections = bool(draw_raw_detections)
        self._trails: dict[int, deque[tuple[int, int, bool]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        if self.draw_raw_detections:
            for detection in result.ball_detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

        active_ids = set()
        for track in result.ball_tracks:
            if track.track_id is None or track.center is None:
                continue
            active_ids.add(track.track_id)
            point = (int(round(track.center[0])), int(round(track.center[1])))
            self._trails[track.track_id].append((*point, track.status == "predicted"))
            self._draw_trail(frame, track.track_id)
            self._draw_track(frame, track)

        for track_id in set(self._trails) - active_ids:
            del self._trails[track_id]

        self._draw_status(frame, result)
        return frame

    def reset(self) -> None:
        self._trails.clear()

    def _draw_trail(self, frame: np.ndarray, track_id: int) -> None:
        points = list(self._trails[track_id])
        colour = _PALETTE[(track_id - 1) % len(_PALETTE)]
        for index in range(1, len(points)):
            x1, y1, _ = points[index - 1]
            x2, y2, _ = points[index]
            cv2.line(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

    def _draw_track(self, frame: np.ndarray, track: BallTrack) -> None:
        track_id = track.track_id or 0
        colour = _PALETTE[(track_id - 1) % len(_PALETTE)]
        cx, cy = int(round(track.center[0])), int(round(track.center[1]))
        if track.bbox:
            x1, y1, x2, y2 = (int(round(value)) for value in track.bbox)
            radius = max(5, min(15, int(max(x2 - x1, y2 - y1) / 2)))
        else:
            radius = 7
        if track.status == "observed":
            cv2.circle(frame, (cx, cy), radius, colour, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, colour, -1, cv2.LINE_AA)
            label = f"ID {track_id} {track.confidence:.0%}"
        else:
            cv2.circle(frame, (cx, cy), radius, colour, 1, cv2.LINE_AA)
            cv2.drawMarker(frame, (cx, cy), colour, cv2.MARKER_CROSS, 8, 1)
            label = f"ID {track_id} pred {track.missing_frames}f"
        cv2.putText(
            frame,
            label,
            (cx + radius + 5, max(16, cy - radius - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_status(frame: np.ndarray, result: FrameResult) -> None:
        observed = sum(track.status == "observed" for track in result.ball_tracks)
        predicted = sum(track.status == "predicted" for track in result.ball_tracks)
        detector_ms = result.diagnostics.get("detector_ms", 0.0)
        text = (
            f"frame {result.frame_index} | tracks {len(result.ball_tracks)} "
            f"(obs {observed}, pred {predicted}) | detector {detector_ms:.1f} ms"
        )
        overlay = frame[:30].copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (24, 24, 24), -1)
        frame[:30] = cv2.addWeighted(overlay, 0.25, frame[:30], 0.75, 0)
        cv2.putText(
            frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (235, 235, 235), 1, cv2.LINE_AA,
        )
