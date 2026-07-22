"""
Simple image-space ball tracker — Kalman + distance gate + smoothing.

Only three ball states::

    observed   — detection accepted this frame
    predicted  — no detection, short-term Kalman predict
    absent     — no reliable ball position

Logic per frame
---------------
1. If no history → accept highest-confidence detection (≥ min_conf).
2. If history exists → pick closest detection within max_jump_px of
   the Kalman-predicted position.
3. No gated detection → predict up to max_predict_frames frames.
4. Beyond max_predict_frames → status = "absent".
5. Reacquire: first detection after absent uses a wider gate
   (max_jump_px * 2) to allow the ball to have moved during absence.
6. Apply moving-average smoothing over the last N observed positions.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .types import BallDetection, BallTrack


# ---------------------------------------------------------------------------
# Kalman core — 4-state constant-velocity, pure numpy
# ---------------------------------------------------------------------------

class _ConstantVelocityKalman:
    """4-state constant-velocity Kalman filter (x, y, vx, vy)."""

    def __init__(self, process_noise: float = 15.0,
                 measurement_noise: float = 2.0):
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._initialised = False

        dt = 1.0
        q = process_noise ** 2
        r = measurement_noise ** 2

        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self._Q = np.array([
            [q * dt**4 / 4, 0,             q * dt**3 / 2, 0            ],
            [0,             q * dt**4 / 4, 0,             q * dt**3 / 2],
            [q * dt**3 / 2, 0,             q * dt**2,     0            ],
            [0,             q * dt**3 / 2, 0,             q * dt**2    ],
        ], dtype=np.float64)
        self._R = np.eye(2, dtype=np.float64) * r

    @property
    def initialised(self) -> bool:
        return self._initialised

    def init(self, x: float, y: float) -> None:
        self._x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 100.0
        self._initialised = True

    def predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, mx: float, my: float) -> None:
        if not self._initialised:
            self.init(mx, my)
            return
        z = np.array([[mx], [my]], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P

    @property
    def position(self) -> tuple[float, float]:
        return (float(self._x[0, 0]), float(self._x[1, 0]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self._x[2, 0]), float(self._x[3, 0]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_roi(cx: float, cy: float, radius: float,
              fw: int, fh: int, min_size: float) -> list[float]:
    x1 = max(0.0, cx - radius)
    y1 = max(0.0, cy - radius)
    x2 = min(float(fw), cx + radius)
    y2 = min(float(fh), cy + radius)
    if x2 - x1 < min_size:
        mid = (x1 + x2) / 2
        x1 = max(0.0, mid - min_size / 2)
        x2 = min(float(fw), mid + min_size / 2)
    if y2 - y1 < min_size:
        mid = (y1 + y2) / 2
        y1 = max(0.0, mid - min_size / 2)
        y2 = min(float(fh), mid + min_size / 2)
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


# ---------------------------------------------------------------------------
# Simple Ball Tracker
# ---------------------------------------------------------------------------

class SimpleBallTracker:
    """Simple Kalman tracker with one distance gate and position smoothing.

    Parameters
    ----------
    min_conf : float
        Minimum detection confidence (single threshold, no tiers).
    max_jump_px : float
        Maximum allowed distance between a detection and the predicted
        position.  Detections farther away are ignored.
    max_predict_frames : int
        How many consecutive frames to predict before marking absent.
    smoothing_window : int
        Number of recent observed positions to average for the final
        reported center.  Set to 1 to disable smoothing.
    process_noise : float
        Kalman process noise (higher = more responsive to detections).
    measurement_noise : float
        Kalman measurement noise (higher = smoother but slower).
    """

    def __init__(
        self,
        min_conf: float = 0.25,
        max_jump_px: float = 180.0,
        max_predict_frames: int = 3,
        smoothing_window: int = 3,
        process_noise: float = 15.0,
        measurement_noise: float = 2.0,
    ):
        self.min_conf = min_conf
        self.max_jump_px = max_jump_px
        self.max_predict_frames = max_predict_frames
        self.smoothing_window = max(1, smoothing_window)

        self._kf = _ConstantVelocityKalman(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )
        self._has_history: bool = False
        self._missing_frames: int = 0
        self._confidence: float = 0.0
        self._last_roi: Optional[list[float]] = None
        self._last_bbox: Optional[list[float]] = None
        self._last_drop_reason: Optional[str] = None
        self._position_history: deque[tuple[float, float]] = deque(
            maxlen=self.smoothing_window
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_roi(self) -> Optional[list[float]]:
        """Suggested search region for the next frame, or None."""
        return self._last_roi

    @property
    def drop_reason(self) -> Optional[str]:
        """Why the last detection was dropped, or None if accepted."""
        return self._last_drop_reason

    def update(
        self,
        detections: list[BallDetection],
        frame_width: int,
        frame_height: int,
    ) -> BallTrack:
        """Advance the tracker by one frame.

        Args:
            detections: Ball detections for this frame (may be empty).
            frame_width, frame_height: Image dimensions in pixels.

        Returns:
            ``BallTrack`` with current status and position.
        """
        best = self._select_detection(detections)

        if best is not None:
            return self._handle_hit(best, frame_width, frame_height)
        else:
            return self._handle_miss(frame_width, frame_height)

    def reset(self) -> None:
        """Reset all internal state."""
        self._kf = _ConstantVelocityKalman()
        self._has_history = False
        self._missing_frames = 0
        self._confidence = 0.0
        self._last_roi = None
        self._last_bbox = None
        self._position_history.clear()

    # ------------------------------------------------------------------
    # Detection selection
    # ------------------------------------------------------------------

    def _select_detection(
        self, detections: list[BallDetection]
    ) -> Optional[BallDetection]:
        """Return the best detection passing the confidence + distance gates."""
        if not detections:
            self._last_drop_reason = "no_detection"
            return None

        # 1. Confidence gate — single threshold, no tiers
        candidates = [d for d in detections if d.confidence >= self.min_conf]
        if not candidates:
            self._last_drop_reason = "low_conf"
            return None

        # 2. If no history, accept the highest-confidence detection
        if not self._has_history or not self._kf.initialised:
            self._last_drop_reason = None
            return max(candidates, key=lambda d: d.confidence)

        # 3. Has history — pick the closest detection within max_jump_px
        px, py = self._kf.position

        best: Optional[BallDetection] = None
        best_dist = float("inf")
        for d in candidates:
            dx = d.center[0] - px
            dy = d.center[1] - py
            dist = np.hypot(dx, dy)
            if dist > self.max_jump_px:
                continue
            if dist < best_dist:
                best_dist = dist
                best = d

        if best is None:
            self._last_drop_reason = "large_jump"
        else:
            self._last_drop_reason = None

        return best

    # ------------------------------------------------------------------
    # Hit / Miss handlers
    # ------------------------------------------------------------------

    def _handle_hit(
        self, det: BallDetection, fw: int, fh: int
    ) -> BallTrack:
        """Accept a detection — Kalman update + smoothing."""
        cx, cy = det.center[0], det.center[1]

        # Kalman: predict then update
        if self._has_history:
            self._kf.predict()
        self._kf.update(cx, cy)

        self._has_history = True
        self._missing_frames = 0
        self._confidence = det.confidence
        self._last_bbox = list(det.bbox)

        # Smoothing: store raw Kalman position, report smoothed
        kx, ky = self._kf.position
        self._position_history.append((kx, ky))

        smooth_x, smooth_y = self._smoothed_position()

        self._last_roi = _clip_roi(smooth_x, smooth_y, 120.0, fw, fh, 60.0)

        return BallTrack(
            status="observed",
            center=[smooth_x, smooth_y],
            bbox=self._last_bbox,
            confidence=self._confidence,
            velocity=list(self._kf.velocity),
            missing_frames=0,
            source="detector",
            roi=self._last_roi,
        )

    def _handle_miss(self, fw: int, fh: int) -> BallTrack:
        """No gated detection — predict or go absent."""
        if not self._has_history:
            return BallTrack(
                status="absent",
                confidence=0.0,
                missing_frames=0,
                source="none",
            )

        self._missing_frames += 1

        if self._missing_frames > self.max_predict_frames:
            # Too long — enter absent, reset history so next detection
            # can reacquire without being gated against a stale position.
            self._last_roi = None
            self._has_history = False
            self._position_history.clear()
            return BallTrack(
                status="absent",
                center=None,
                confidence=0.0,
                missing_frames=self._missing_frames,
                velocity=None,
                source="none",
            )

        # Short-term prediction
        self._kf.predict()
        px, py = self._kf.position

        # Smooth using recent history (no new observation added)
        self._position_history.append((px, py))
        smooth_x, smooth_y = self._smoothed_position()

        decayed_conf = self._confidence * max(
            0.0, 1.0 - self._missing_frames / (self.max_predict_frames + 1)
        )

        return BallTrack(
            status="predicted",
            center=[smooth_x, smooth_y],
            confidence=decayed_conf,
            velocity=list(self._kf.velocity),
            missing_frames=self._missing_frames,
            source="prediction",
            roi=None,
        )

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _smoothed_position(self) -> tuple[float, float]:
        """Return the moving-average of recent positions."""
        if not self._position_history:
            kx, ky = self._kf.position
            return (kx, ky)
        xs = [p[0] for p in self._position_history]
        ys = [p[1] for p in self._position_history]
        return (float(np.mean(xs)), float(np.mean(ys)))
