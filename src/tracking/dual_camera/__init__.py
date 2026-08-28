"""Synchronized dual-camera tracking orchestration."""

from .coordinator import (
    CrossCameraHandoffAdvisor,
    CrossHalfBallCoordinator,
    GlobalBallSelection,
)

__all__ = [
    "CrossCameraHandoffAdvisor",
    "CrossHalfBallCoordinator",
    "GlobalBallSelection",
]
