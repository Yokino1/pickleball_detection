"""Configuration assembly for the optional downstream referee engine."""

from __future__ import annotations

from ..court.layout import CourtLayout
from .rally import RallyRefereeEngine
from .types import REFEREE_CONTRACT_VERSION


def build_referee_engine(
    config: dict,
    layout: CourtLayout | None,
) -> RallyRefereeEngine | None:
    """Build the referee from ``runtime.referee`` when explicitly enabled."""
    values = dict(config.get("runtime", {}).get("referee", {}))
    if not values.get("enabled", False):
        return None
    if layout is None:
        return None
    contract_version = int(
        values.get("contract_version", REFEREE_CONTRACT_VERSION)
    )
    if contract_version != REFEREE_CONTRACT_VERSION:
        raise ValueError(
            "unsupported referee contract_version="
            f"{contract_version}; expected {REFEREE_CONTRACT_VERSION}"
        )
    side_mapping = dict(values.get("side_mapping", {}))
    court_values = dict(
        config.get("runtime", {}).get("court_projection", {})
    )
    return RallyRefereeEngine(
        layout,
        enabled=True,
        low_y_side=side_mapping.get("low_y", values.get("low_y_side", "left")),
        high_y_side=side_mapping.get(
            "high_y",
            values.get("high_y_side", "right"),
        ),
        scoring_mode=values.get("scoring_mode", "rally_point"),
        initial_score=values.get("initial_score"),
        # ``runtime.referee.net_deadband_ft`` remains a compatibility override
        # for older configs.  The maintained profile owns this geometry under
        # court_projection so events and referee share one physical-half rule.
        net_deadband_ft=values.get(
            "net_deadband_ft",
            court_values.get("net_deadband_ft", 0.75),
        ),
        route_confirm_observed_frames=values.get("route_confirm_observed_frames", 3),
        route_min_netward_displacement_ft=values.get("route_min_netward_displacement_ft", 1.0),
        net_cross_confirm_observed_frames=values.get("net_cross_confirm_observed_frames", 2),
        net_cross_min_displacement_ft=values.get("net_cross_min_displacement_ft", 0.15),
        max_observation_gap_ms=values.get("max_observation_gap_ms", 120.0),
        unavailable_timeout_ms=values.get(
            "rally_lost_timeout_ms",
            values.get("unavailable_timeout_ms", 5000.0),
        ),
        post_rally_guard_ms=values.get("post_rally_guard_ms", 800.0),
        post_rally_rearm_missing_ms=values.get(
            "post_rally_rearm_missing_ms",
            200.0,
        ),
        post_rally_rearm_stable_observed_frames=values.get(
            "post_rally_rearm_stable_observed_frames",
            5,
        ),
        post_rally_rearm_stable_displacement_ft=values.get(
            "post_rally_rearm_stable_displacement_ft",
            0.5,
        ),
        hit_assist_window_ms=values.get("hit_assist_window_ms", 300.0),
        observation_margin_ft=values.get("observation_margin_ft", 6.0),
        motion_confirmation_window_ms=values.get(
            "motion_confirmation_window_ms"
        ),
        pending_bounce_timeout_ms=values.get("pending_bounce_timeout_ms"),
        require_serve_arming=values.get("require_serve_arming", True),
        expected_server_side=values.get("expected_server_side"),
        service_ownership_enabled=values.get(
            "service_ownership_enabled",
            False,
        ),
        initial_server_side=values.get("initial_server_side"),
        require_server_player_center_outside_court=values.get(
            "require_server_player_center_outside_court",
            False,
        ),
    )
