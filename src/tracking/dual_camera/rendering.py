"""Side-by-side rendering for synchronized dual-camera tracking."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .coordinator import GlobalBallSelection

HEADER_HEIGHT = 82


def display_result(result, selected_track):
    tracks = [selected_track] if selected_track is not None else []
    return replace(
        result,
        ball_tracks=tracks,
        ball_track=selected_track,
    )


def court_motion_discontinuity_reason(
    trail_reset_reason: str | None,
) -> str | None:
    """Map display reset reasons to physical court-motion discontinuities."""
    if trail_reset_reason in {"local_track_change", "physical_discontinuity"}:
        return trail_reset_reason
    return None


def make_header(
    width: int,
    left_width: int,
    selection: GlobalBallSelection,
    court_panel_width: int = 0,
) -> np.ndarray:
    header = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
    tracking_width = max(left_width, width - max(0, int(court_panel_width)))
    global_text = (
        f"GLOBAL BALL ID {selection.global_track_id}: "
        f"{selection.state.upper()}"
    )

    def put_centered(
        text: str,
        x1: int,
        x2: int,
        baseline_y: int,
        scale: float,
    ) -> None:
        text_width = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            2,
        )[0][0]
        text_x = max(x1 + 8, x1 + (x2 - x1 - text_width) // 2)
        cv2.putText(
            header,
            text,
            (text_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    put_centered(global_text, 0, tracking_width, 30, 0.72)
    put_centered("LEFT HALF", 0, left_width, 67, 0.78)
    put_centered("RIGHT HALF", left_width, tracking_width, 67, 0.78)
    if court_panel_width > 0:
        put_centered(
            "COURT PROJECTION",
            tracking_width,
            width,
            51,
            0.58,
        )
    return header
