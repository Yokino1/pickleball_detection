"""
Court zone classification — pure geometry, no image dependency.

Divides the 20ft × 44ft pickleball court into zones based on
standard court coordinates (feet).  Always available, always
deterministic.

Zones (from far to near)::

    far_left_service   far_right_service
    ┌────────┬────────┐  ← far baseline (y=0)
    │   L    │   R    │
    │        │        │
    ├────────┼────────┤  ← far kitchen line (y=15)
    │        │        │
    │  far_nvz (NVZ)  │  ← net (y=22)
    │        │        │
    ├────────┼────────┤
    │        │        │
    │ near_nvz (NVZ)  │  ← near kitchen line (y=29)
    │        │        │
    ├────────┼────────┤
    │   L    │   R    │
    │        │        │
    └────────┴────────┘  ← near baseline (y=44)
    near_left_service  near_right_service

Output is always a weak label — not a formal line call.
"""

from __future__ import annotations

from typing import Optional

from .layout import (
    COURT_WIDTH_FT,
    COURT_LENGTH_FT,
    FAR_BASELINE_Y,
    FAR_KITCHEN_Y,
    NET_Y,
    NEAR_KITCHEN_Y,
    NEAR_BASELINE_Y,
    CENTER_X,
)

# Margin (ft) for "out" vs "on boundary" — points within this distance
# of a zone boundary are labelled as the zone, not as out.
BOUNDARY_MARGIN_FT = 0.1


def classify_ball_zone(
    ball_court_xy: Optional[tuple[float, float]],
) -> dict:
    """Classify a ball position (in court feet) into a zone.

    Args:
        ball_court_xy: (x, y) in court feet, or None.

    Returns:
        Dict with:
        - ``zone``: zone label string
        - ``is_in_court``: True if within the 20×44 bounds
        - ``is_nvz``: True if inside either non-volley zone
    """
    if ball_court_xy is None:
        return {"zone": "unknown", "is_in_court": False, "is_nvz": False}

    x, y = ball_court_xy

    # Out of court
    if x < -BOUNDARY_MARGIN_FT or x > COURT_WIDTH_FT + BOUNDARY_MARGIN_FT:
        return {"zone": "out_of_court", "is_in_court": False, "is_nvz": False}
    if y < -BOUNDARY_MARGIN_FT or y > COURT_LENGTH_FT + BOUNDARY_MARGIN_FT:
        return {"zone": "out_of_court", "is_in_court": False, "is_nvz": False}

    # Clamp to court bounds for zone lookup
    x_c = max(0, min(COURT_WIDTH_FT, x))
    y_c = max(0, min(COURT_LENGTH_FT, y))
    is_left = x_c < CENTER_X

    if y_c < FAR_KITCHEN_Y:
        zone = "far_left_service" if is_left else "far_right_service"
        nvz = False
    elif y_c < NET_Y:
        zone = "far_nvz"
        nvz = True
    elif y_c < NEAR_KITCHEN_Y:
        zone = "near_nvz"
        nvz = True
    else:
        zone = "near_left_service" if is_left else "near_right_service"
        nvz = False

    return {"zone": zone, "is_in_court": True, "is_nvz": nvz}


# Human-readable labels
ZONE_LABELS = {
    "far_left_service":   "Far Left Service",
    "far_right_service":  "Far Right Service",
    "far_nvz":            "Far NVZ (Kitchen)",
    "near_nvz":           "Near NVZ (Kitchen)",
    "near_left_service":  "Near Left Service",
    "near_right_service": "Near Right Service",
    "out_of_court":       "Out of Court",
    "unknown":            "Unknown",
}
