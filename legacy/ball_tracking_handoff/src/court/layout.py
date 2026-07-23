"""
Standard pickleball court geometry — 20ft × 44ft, 14 keypoints.

This module defines the canonical court layout in court coordinates (feet).
It has **no dependency** on images, detectors, or calibration — the layout
is always available and can always be drawn.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Court dimensions (feet)
# ---------------------------------------------------------------------------

COURT_WIDTH_FT = 20.0   # sideline to sideline
COURT_LENGTH_FT = 44.0  # baseline to baseline
KITCHEN_DEPTH_FT = 7.0  # non-volley zone depth from net

NET_Y = 22.0             # net line (halfway)
FAR_BASELINE_Y = 0.0     # far baseline
NEAR_BASELINE_Y = 44.0   # near baseline
FAR_KITCHEN_Y = 15.0     # far kitchen (NVZ) line (22 - 7 = 15)
NEAR_KITCHEN_Y = 29.0    # near kitchen (NVZ) line (22 + 7 = 29)
CENTER_X = 10.0          # center service line x

# ---------------------------------------------------------------------------
# 14-point schema
# ---------------------------------------------------------------------------
# Layout (top-down view, far side = top):
#
#      0───────1───────2       Far Baseline (left / center / right)
#      │               │
#      3       4       5       Far Kitchen Line (left / center / right)
#      │               │
#      ├───────6─7──────┤      Net endpoints (left / right)
#      │               │
#      8       9      10       Near Kitchen Line (left / center / right)
#      │               │
#     11──────12──────13       Near Baseline (left / center / right)

# Keypoint index constants
FAR_BASELINE_LEFT = 0
FAR_BASELINE_CENTER = 1
FAR_BASELINE_RIGHT = 2
FAR_KITCHEN_LEFT = 3
FAR_KITCHEN_CENTER = 4
FAR_KITCHEN_RIGHT = 5
NET_LEFT = 6
NET_RIGHT = 7
NEAR_KITCHEN_LEFT = 8
NEAR_KITCHEN_CENTER = 9
NEAR_KITCHEN_RIGHT = 10
NEAR_BASELINE_LEFT = 11
NEAR_BASELINE_CENTER = 12
NEAR_BASELINE_RIGHT = 13

NUM_KEYPOINTS = 14

# Canonical keypoint coordinates in court feet: (x, y)
# Origin (0, 0) = far-baseline-left corner
CANONICAL_KEYPOINTS: dict[int, tuple[float, float]] = {
    0:  (0.0, 0.0),           # far_baseline_left
    1:  (10.0, 0.0),          # far_baseline_center
    2:  (20.0, 0.0),          # far_baseline_right
    3:  (0.0, 15.0),          # far_kitchen_left
    4:  (10.0, 15.0),         # far_kitchen_center
    5:  (20.0, 15.0),         # far_kitchen_right
    6:  (0.0, 22.0),          # net_left
    7:  (20.0, 22.0),         # net_right
    8:  (0.0, 29.0),          # near_kitchen_left
    9:  (10.0, 29.0),         # near_kitchen_center
    10: (20.0, 29.0),         # near_kitchen_right
    11: (0.0, 44.0),          # near_baseline_left
    12: (10.0, 44.0),         # near_baseline_center
    13: (20.0, 44.0),         # near_baseline_right
}

# Human-readable names
KEYPOINT_NAMES: dict[int, str] = {
    0: "far_baseline_left",
    1: "far_baseline_center",
    2: "far_baseline_right",
    3: "far_kitchen_left",
    4: "far_kitchen_center",
    5: "far_kitchen_right",
    6: "net_left",
    7: "net_right",
    8: "near_kitchen_left",
    9: "near_kitchen_center",
    10: "near_kitchen_right",
    11: "near_baseline_left",
    12: "near_baseline_center",
    13: "near_baseline_right",
}

# Court line segments — pairs of keypoint indices
COURT_LINE_SEGMENTS: list[tuple[int, int]] = [
    # Outer boundary
    (0, 2),    # Far baseline
    (11, 13),  # Near baseline
    (0, 11),   # Left sideline
    (2, 13),   # Right sideline
    # Net
    (6, 7),    # Net
    # Kitchen (NVZ) lines
    (3, 5),    # Far kitchen line
    (8, 10),   # Near kitchen line
    # Center service lines (baseline → kitchen, not through NVZ)
    (1, 4),    # Far center service line
    (9, 12),   # Near center service line
    # Kitchen boundaries (vertical)
    (3, 8),    # Left kitchen boundary
    (5, 10),   # Right kitchen boundary
]

COURT_LINE_SEGMENTS_TOPDOWN: list[tuple[int, int]] = list(COURT_LINE_SEGMENTS)

# Side-view court line segments, split at every semantic keypoint.
#
# The 14 canonical points keep the same physical court coordinates as before:
# x = court width (0..20 ft), y = court length (0..44 ft). The side-view
# drawing reads the y/length axis left-to-right:
#
#   kp2--kp5--kp7--kp10--kp13
#    |    |    |     |     |
#   kp1--kp4         kp9--kp12
#    |    |    |     |     |
#   kp0--kp3--kp6---kp8--kp11
COURT_LINE_SEGMENTS_SIDEVIEW: list[tuple[int, int]] = [
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 7),
    (8, 9), (9, 10),
    (11, 12), (12, 13),
    (2, 5), (5, 7), (7, 10), (10, 13),
    (1, 4), (9, 12),
    (0, 3), (3, 6), (6, 8), (8, 11),
]

COURT_LINE_SEGMENTS_SHORT_END: list[tuple[int, int]] = list(COURT_LINE_SEGMENTS_SIDEVIEW)
COURT_LINE_SEGMENTS = list(COURT_LINE_SEGMENTS_SIDEVIEW)

SIDEVIEW_KEYPOINT_NAMES: dict[int, str] = {
    0: "left_baseline_lower",
    1: "left_baseline_center",
    2: "left_baseline_upper",
    3: "left_kitchen_lower",
    4: "left_kitchen_center",
    5: "left_kitchen_upper",
    6: "net_lower",
    7: "net_upper",
    8: "right_kitchen_lower",
    9: "right_kitchen_center",
    10: "right_kitchen_upper",
    11: "right_baseline_lower",
    12: "right_baseline_center",
    13: "right_baseline_upper",
}


# ---------------------------------------------------------------------------
# CourtLayout class
# ---------------------------------------------------------------------------

class CourtLayout:
    """Standard pickleball court geometry.

    Always available — does not depend on detection results.
    """

    def __init__(self, view: str = "sideview"):
        self.view = view

    # -- dimensions -------------------------------------------------------

    @property
    def width_ft(self) -> float:
        return COURT_WIDTH_FT

    @property
    def length_ft(self) -> float:
        return COURT_LENGTH_FT

    @property
    def net_y(self) -> float:
        return NET_Y

    @property
    def far_kitchen_y(self) -> float:
        return FAR_KITCHEN_Y

    @property
    def near_kitchen_y(self) -> float:
        return NEAR_KITCHEN_Y

    # -- keypoints --------------------------------------------------------

    @property
    def canonical_keypoints(self) -> dict[int, tuple[float, float]]:
        """Return all 14 keypoints in court feet coordinates."""
        return dict(CANONICAL_KEYPOINTS)

    @property
    def keypoint_names(self) -> dict[int, str]:
        if self.view == "sideview":
            return dict(SIDEVIEW_KEYPOINT_NAMES)
        return dict(KEYPOINT_NAMES)

    @property
    def line_segments(self) -> list[tuple[int, int]]:
        """Return court line segments as (start_kp_idx, end_kp_idx) pairs."""
        return self.line_segments_for_view(self.view)

    @staticmethod
    def line_segments_for_view(view: str = "sideview") -> list[tuple[int, int]]:
        if view == "topdown":
            return list(COURT_LINE_SEGMENTS_TOPDOWN)
        if view == "short_end":
            return list(COURT_LINE_SEGMENTS_SHORT_END)
        return list(COURT_LINE_SEGMENTS_SIDEVIEW)

    # -- mini-court drawing -----------------------------------------------

    def mini_court_points(
        self, width_px: int, height_px: int
    ) -> dict:
        """Compute pixel coordinates for drawing a mini-court.

        Args:
            width_px: Available drawing width in pixels.
            height_px: Available drawing height in pixels.

        Returns:
            Dict with:
            - ``keypoints_px``: list of 14 (x, y) pixel tuples
            - ``lines``: list of (start_xy, end_xy) pixel tuples
            - ``padding_x``, ``padding_y``: margins used
        """
        padding = 10  # px margin inside the drawing area
        draw_w = width_px - 2 * padding
        draw_h = height_px - 2 * padding

        view = self.view
        display_w_ft = COURT_LENGTH_FT if view == "sideview" else COURT_WIDTH_FT
        display_h_ft = COURT_WIDTH_FT if view == "sideview" else COURT_LENGTH_FT

        # Maintain the selected reference-view aspect ratio.
        aspect = display_w_ft / display_h_ft
        if draw_w / draw_h > aspect:
            # Height-constrained
            actual_h = draw_h
            actual_w = int(draw_h * aspect)
        else:
            # Width-constrained
            actual_w = draw_w
            actual_h = int(draw_w / aspect)

        offset_x = padding + (draw_w - actual_w) // 2
        offset_y = padding + (draw_h - actual_h) // 2

        px_per_ft_x = actual_w / display_w_ft
        px_per_ft_y = actual_h / display_h_ft

        def _to_px(ft_x: float, ft_y: float) -> tuple[int, int]:
            if view == "sideview":
                display_x = ft_y
                display_y = COURT_WIDTH_FT - ft_x
            else:
                display_x = ft_x
                display_y = ft_y
            return (
                int(offset_x + display_x * px_per_ft_x),
                int(offset_y + display_y * px_per_ft_y),
            )

        keypoints_px = [_to_px(x, y) for x, y in CANONICAL_KEYPOINTS.values()]

        lines_px = []
        for i, j in COURT_LINE_SEGMENTS:
            lines_px.append((keypoints_px[i], keypoints_px[j]))

        return {
            "keypoints_px": keypoints_px,
            "lines": lines_px,
            "origin_px": (offset_x, offset_y),
            "size_px": (actual_w, actual_h),
            "px_per_ft": (px_per_ft_x, px_per_ft_y),
            "view": view,
        }

    def court_ft_to_mini_px(
        self, court_xy: tuple[float, float], mini_info: dict
    ) -> tuple[int, int]:
        """Convert a court-foot position to mini-court pixel coordinates."""
        ft_x, ft_y = court_xy
        ox, oy = mini_info["origin_px"]
        pfx, pfy = mini_info["px_per_ft"]
        if mini_info.get("view") == "sideview":
            display_x = ft_y
            display_y = COURT_WIDTH_FT - ft_x
        else:
            display_x = ft_x
            display_y = ft_y
        return (int(ox + display_x * pfx), int(oy + display_y * pfy))
