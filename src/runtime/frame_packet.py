"""Timestamped frame contracts shared by capture and synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


VALID_SIDES = ("left", "right")


@dataclass(frozen=True)
class FramePacket:
    """One captured frame with its source-side identity and capture timestamp."""

    side: str
    sequence: int
    capture_timestamp_s: float
    frame: object
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError("FramePacket.side must be 'left' or 'right'")
        if int(self.sequence) < 0:
            raise ValueError("FramePacket.sequence must be non-negative")
        if not isfinite(float(self.capture_timestamp_s)):
            raise ValueError(
                "FramePacket.capture_timestamp_s must be finite"
            )


@dataclass(frozen=True)
class FramePair:
    """A left/right frame pair selected by capture-time proximity."""

    left: FramePacket
    right: FramePacket

    def __post_init__(self) -> None:
        if self.left.side != "left" or self.right.side != "right":
            raise ValueError("FramePair requires left and right packets")

    @property
    def timestamp_s(self) -> float:
        return (
            float(self.left.capture_timestamp_s)
            + float(self.right.capture_timestamp_s)
        ) / 2.0

    @property
    def skew_s(self) -> float:
        return (
            float(self.right.capture_timestamp_s)
            - float(self.left.capture_timestamp_s)
        )

    @property
    def skew_ms(self) -> float:
        return self.skew_s * 1000.0
