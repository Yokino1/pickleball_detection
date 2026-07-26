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


def make_header(
    width: int,
    left_width: int,
    selection: GlobalBallSelection,
) -> np.ndarray:
    header = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
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

    put_centered(global_text, 0, width, 30, 0.72)
    put_centered("LEFT HALF", 0, left_width, 67, 0.78)
    put_centered("RIGHT HALF", left_width, width, 67, 0.78)
    return header
