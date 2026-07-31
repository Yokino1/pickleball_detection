"""Public assembly boundary for fixed-camera court projection."""

from __future__ import annotations

from .calibration import load_camera_calibrations
from .events import CourtEventInterpreter
from .layout import CourtLayout
from .projector import FixedCourtProjector
from .renderer import CourtPanelRenderer


def build_court_projection(
    config: dict,
    frame_sizes: dict[str, tuple[int, int]],
) -> tuple[
    FixedCourtProjector | None,
    CourtPanelRenderer | None,
    CourtEventInterpreter | None,
]:
    """Build the read-only projector, renderer and event interpreter."""
    values = dict(
        config.get("runtime", {}).get("court_projection", {})
    )
    if not values.get("enabled", False):
        return None, None, None
    layout = CourtLayout(
        coordinate_system=str(
            values.get(
                "coordinate_system",
                "pickleball_full_court_ft",
            )
        ),
        coordinate_system_version=int(
            values.get("coordinate_system_version", 1)
        ),
    )
    calibrations = load_camera_calibrations(
        values,
        frame_sizes,
        layout,
    )
    projector = FixedCourtProjector(calibrations, layout)
    renderer = CourtPanelRenderer(
        layout,
        preferred_width=values.get("panel_width", 560),
        margin_px=values.get("panel_margin_px", 28),
        outside_margin_ft=values.get("outside_margin_ft", 30.0),
        trail_length=values.get("trail_length", 15),
        status_font_path=values.get("status_font_path"),
        status_font_size=values.get("status_font_size", 48),
    )
    event_values = dict(values.get("event_interpretation", {}))
    event_interpreter = CourtEventInterpreter(
        enabled=event_values.get("enabled", True),
        hit_flash_frames=event_values.get("hit_flash_frames", 3),
        ground_min_hold_frames=event_values.get(
            "ground_min_hold_frames",
            3,
        ),
        rise_confirm_frames=event_values.get("rise_confirm_frames", 2),
        rise_min_speed_px_per_second=event_values.get(
            "rise_min_speed_px_per_second",
            40.0,
        ),
        kinematic_enabled=event_values.get("kinematic_enabled", True),
        max_observation_gap_ms=event_values.get(
            "max_observation_gap_ms",
            80.0,
        ),
        bounce_min_downward_speed_px_per_second=event_values.get(
            "bounce_min_downward_speed_px_per_second",
            100.0,
        ),
        bounce_min_upward_speed_px_per_second=event_values.get(
            "bounce_min_upward_speed_px_per_second",
            40.0,
        ),
        bounce_max_horizontal_impulse_ratio=event_values.get(
            "bounce_max_horizontal_impulse_ratio",
            1.5,
        ),
        hit_min_direction_change_deg=event_values.get(
            "hit_min_direction_change_deg",
            35.0,
        ),
        hit_min_impulse_px_per_second=event_values.get(
            "hit_min_impulse_px_per_second",
            300.0,
        ),
        discontinuity_hit_max_speed_px_per_second=event_values.get(
            "discontinuity_hit_max_speed_px_per_second",
            3200.0,
        ),
        event_cooldown_ms=event_values.get("event_cooldown_ms", 160.0),
        post_hit_bounce_suppression_ms=event_values.get(
            "post_hit_bounce_suppression_ms",
            180.0,
        ),
        rally_state_timeout_ms=event_values.get(
            "rally_state_timeout_ms",
            5000.0,
        ),
        player_contact_margin_ratio=event_values.get(
            "player_contact_margin_ratio",
            0.20,
        ),
        player_reach_margin_ratio=event_values.get(
            "player_reach_margin_ratio",
            0.35,
        ),
        player_foot_band_ratio=event_values.get(
            "player_foot_band_ratio",
            0.18,
        ),
    )
    return projector, renderer, event_interpreter
