"""Additive 0811 output vocabulary without changing referee decisions."""

from __future__ import annotations

from typing import Any, Iterable


TERMINAL_EVENT_0811 = {
    "serve_net": "off_net",
    "return_net": "off_net",
    "out_of_bounds": "firstlanding_out",
    "second_bounce": "doublelanding_in",
    "unknown": "UNKNOWN",
}

FRAME_EVENT_0811 = {
    "first_bounce_recorded": "firstlanding_in",
    "first_bounce_inside_unknown": "UNKNOWN",
    "volleyed_return": "volleyed",
    **TERMINAL_EVENT_0811,
}

_EVENT_PRIORITY = {
    "NONE": 0,
    "firstlanding_in": 1,
    "volleyed": 2,
    "off_net": 3,
    "firstlanding_out": 4,
    "doublelanding_in": 5,
    "UNKNOWN": 6,
}


def map_event_type_0811(
    events: Iterable[str],
    *,
    terminal_event: str | None = None,
) -> str:
    """Return one deterministic 0811 label for a frame or terminal result."""
    if terminal_event is not None:
        return TERMINAL_EVENT_0811.get(str(terminal_event), "UNKNOWN")
    mapped = [
        FRAME_EVENT_0811[name]
        for name in (str(item) for item in events)
        if name in FRAME_EVENT_0811
    ]
    if not mapped:
        return "NONE"
    return max(mapped, key=lambda name: _EVENT_PRIORITY[name])


def rally_over_0811(*, phase: str, terminal_status: str | None = None) -> str:
    """Map the one-shot referee state to the 0811 Y/N/UNKNOWN/NONE field."""
    if terminal_status is not None:
        return "UNKNOWN" if terminal_status == "unresolved" else "Y"
    if phase in {"SERVE_CONFIRMING", "IN_RALLY"}:
        return "N"
    return "NONE"


def target_ball_state_0811(projection: Any) -> str:
    """Map existing tracking visibility without conflating ROI exit with OUT."""
    track_status = str(getattr(projection, "track_status", "") or "").lower()
    if bool(getattr(projection, "observed", False)) or track_status == "observed":
        return "target_ball_tracked"
    if bool(getattr(projection, "predicted", False)) or track_status == "predicted":
        return "target_ball_predicted"
    if getattr(projection, "active_side", None) is None or track_status in {
        "",
        "absent",
        "lost",
        "unavailable",
    }:
        return "target_ball_lost"
    return "target_ball_predictfailed"
