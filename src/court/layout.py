"""Canonical 20 x 44 ft pickleball court geometry."""

from __future__ import annotations

from dataclasses import dataclass


COURT_WIDTH_FT = 20.0
COURT_LENGTH_FT = 44.0
NET_Y_FT = 22.0
FAR_KITCHEN_Y_FT = 15.0
NEAR_KITCHEN_Y_FT = 29.0

CANONICAL_KEYPOINTS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (10.0, 0.0),
    2: (20.0, 0.0),
    3: (0.0, 15.0),
    4: (10.0, 15.0),
    5: (20.0, 15.0),
    6: (0.0, 22.0),
    7: (20.0, 22.0),
    8: (0.0, 29.0),
    9: (10.0, 29.0),
    10: (20.0, 29.0),
    11: (0.0, 44.0),
    12: (10.0, 44.0),
    13: (20.0, 44.0),
}

COURT_LINE_SEGMENTS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (11, 13),
    (0, 11),
    (2, 13),
    (3, 5),
    (6, 7),
    (8, 10),
    (1, 4),
    (9, 12),
)


@dataclass(frozen=True)
class CourtLayout:
    """Pure standard-court geometry, independent of images and tracking."""

    coordinate_system: str = "pickleball_full_court_ft"
    coordinate_system_version: int = 1
    width_ft: float = COURT_WIDTH_FT
    length_ft: float = COURT_LENGTH_FT
    net_y_ft: float = NET_Y_FT

    @property
    def canonical_keypoints(self) -> dict[int, tuple[float, float]]:
        return dict(CANONICAL_KEYPOINTS)

    @property
    def line_segments(self) -> tuple[tuple[int, int], ...]:
        return COURT_LINE_SEGMENTS

    def contains(self, court_xy: tuple[float, float]) -> bool:
        x_ft, y_ft = court_xy
        return 0.0 <= x_ft <= self.width_ft and 0.0 <= y_ft <= self.length_ft
