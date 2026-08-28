"""Kalman motion models used by the ball tracker."""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """Small NumPy Kalman filter with state ``x, y, vx, vy``."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.p = np.eye(4, dtype=np.float64) * 100.0
        self.initialized = False
        self.h = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )

    def initialize(self, center: list[float]) -> None:
        self.x[:, 0] = (float(center[0]), float(center[1]), 0.0, 0.0)
        self.p = np.diag(
            [100.0, 100.0, 1_000_000.0, 1_000_000.0]
        )
        self.initialized = True

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
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
        measurement = np.array(
            [[float(center[0])], [float(center[1])]]
        )
        residual = measurement - self.h @ self.x
        innovation = (
            self.h @ self.p @ self.h.T
            + np.eye(2) * self._measurement_noise**2
        )
        gain = self.p @ self.h.T @ np.linalg.inv(innovation)
        self.x = self.x + gain @ residual
        self.p = (np.eye(4) - gain @ self.h) @ self.p

    def innovation_nis(self, center: list[float]) -> float:
        measurement = np.array(
            [[float(center[0])], [float(center[1])]]
        )
        residual = measurement - self.h @ self.x
        innovation = (
            self.h @ self.p @ self.h.T
            + np.eye(2) * self._measurement_noise**2
        )
        return float(
            (residual.T @ np.linalg.solve(innovation, residual)).item()
        )

    def shift_position(self, dx: float, dy: float) -> None:
        self.x[0, 0] += float(dx)
        self.x[1, 0] += float(dy)

    def set_velocity(self, vx: float, vy: float) -> None:
        self.x[2, 0] = float(vx)
        self.x[3, 0] = float(vy)

    def damp_velocity(self, factor: float) -> None:
        self.x[2:, 0] *= float(factor)

    def damp_acceleration(self, factor: float) -> None:
        del factor

    def reset_acceleration(self) -> None:
        return

    def clamp_acceleration(self, maximum: float) -> None:
        del maximum

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def acceleration(self) -> tuple[float, float]:
        return 0.0, 0.0


class ConstantAccelerationKalman:
    """Linear Kalman filter with state ``x, y, vx, vy, ax, ay``."""

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
    ):
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.p = np.eye(6, dtype=np.float64) * 100.0
        self.initialized = False
        self.h = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def initialize(self, center: list[float]) -> None:
        self.x[:, 0] = (
            float(center[0]),
            float(center[1]),
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.p = np.diag(
            [
                100.0,
                100.0,
                1_000_000.0,
                1_000_000.0,
                1_000_000.0,
                1_000_000.0,
            ]
        )
        self.initialized = True

    def predict(self, dt: float = 1.0) -> tuple[float, float]:
        dt2 = dt * dt
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = self._process_noise**2
        axis_q = np.array(
            [
                [dt**6 / 36.0, dt**5 / 12.0, dt**4 / 6.0],
                [dt**5 / 12.0, dt**4 / 4.0, dt**3 / 2.0],
                [dt**4 / 6.0, dt**3 / 2.0, dt2],
            ],
            dtype=np.float64,
        ) * q
        q_matrix = np.zeros((6, 6), dtype=np.float64)
        indices_x = np.ix_([0, 2, 4], [0, 2, 4])
        indices_y = np.ix_([1, 3, 5], [1, 3, 5])
        q_matrix[indices_x] = axis_q
        q_matrix[indices_y] = axis_q
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q_matrix
        return self.position

    def update(self, center: list[float]) -> None:
        if not self.initialized:
            self.initialize(center)
            return
        measurement = np.array(
            [[float(center[0])], [float(center[1])]]
        )
        residual = measurement - self.h @ self.x
        innovation = (
            self.h @ self.p @ self.h.T
            + np.eye(2) * self._measurement_noise**2
        )
        gain = self.p @ self.h.T @ np.linalg.inv(innovation)
        self.x = self.x + gain @ residual
        self.p = (np.eye(6) - gain @ self.h) @ self.p

    def innovation_nis(self, center: list[float]) -> float:
        measurement = np.array(
            [[float(center[0])], [float(center[1])]]
        )
        residual = measurement - self.h @ self.x
        innovation = (
            self.h @ self.p @ self.h.T
            + np.eye(2) * self._measurement_noise**2
        )
        return float(
            (residual.T @ np.linalg.solve(innovation, residual)).item()
        )

    def shift_position(self, dx: float, dy: float) -> None:
        self.x[0, 0] += float(dx)
        self.x[1, 0] += float(dy)

    def set_velocity(self, vx: float, vy: float) -> None:
        self.x[2, 0] = float(vx)
        self.x[3, 0] = float(vy)

    def damp_velocity(self, factor: float) -> None:
        self.x[2:4, 0] *= float(factor)

    def damp_acceleration(self, factor: float) -> None:
        self.x[4:6, 0] *= float(factor)

    def reset_acceleration(self) -> None:
        self.x[4:6, 0] = 0.0

    def clamp_acceleration(self, maximum: float) -> None:
        maximum = max(0.0, float(maximum))
        magnitude = float(np.hypot(*self.acceleration))
        if maximum > 0.0 and magnitude > maximum:
            self.x[4:6, 0] *= maximum / magnitude

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def acceleration(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])
