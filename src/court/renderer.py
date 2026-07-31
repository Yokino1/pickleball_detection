"""Render a horizontal blank-court panel and the projected global ball."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .layout import CourtLayout
from .projector import ProjectionResult
from .text import ChineseStatusBanner


class CourtPanelRenderer:
    """Stateful projected-ball trail on a camera-background-free court panel."""

    def __init__(
        self,
        layout: CourtLayout | None = None,
        preferred_width: int = 560,
        margin_px: int = 28,
        outside_margin_ft: float = 30.0,
        trail_length: int = 15,
        status_font_path: str | None = None,
        status_font_size: int = 48,
    ):
        self.layout = layout or CourtLayout()
        self.preferred_width = max(240, int(preferred_width))
        self.margin_px = max(8, int(margin_px))
        self.outside_margin_ft = max(0.0, float(outside_margin_ft))
        self.trail = deque(maxlen=max(1, int(trail_length)))
        self.status_banner = ChineseStatusBanner(
            font_path=status_font_path,
            font_size=status_font_size,
        )

    def panel_width(self, frame_height: int) -> int:
        return min(
            self.preferred_width,
            max(320, int(round(float(frame_height) * 0.5))),
        )

    def reset(self) -> None:
        self.trail.clear()

    def render(
        self,
        frame_height: int,
        projection: ProjectionResult,
    ) -> np.ndarray:
        width = self.panel_width(frame_height)
        panel = np.full((frame_height, width, 3), 24, dtype=np.uint8)
        scale, origin_x, origin_y = self._transform(width, frame_height)

        court_left = int(round(origin_x))
        court_top = int(round(origin_y))
        court_right = int(round(origin_x + self.layout.length_ft * scale))
        court_bottom = int(round(origin_y + self.layout.width_ft * scale))
        cv2.rectangle(
            panel,
            (court_left, court_top),
            (court_right, court_bottom),
            (44, 44, 44),
            cv2.FILLED,
        )

        keypoints = self.layout.canonical_keypoints
        for start_index, end_index in self.layout.line_segments:
            start = self._court_to_panel(
                keypoints[start_index],
                scale,
                origin_x,
                origin_y,
            )
            end = self._court_to_panel(
                keypoints[end_index],
                scale,
                origin_x,
                origin_y,
            )
            color = (
                (255, 160, 0)
                if start_index == 6 and end_index == 7
                else (235, 235, 235)
            )
            thickness = 3 if (start_index, end_index) == (6, 7) else 2
            cv2.line(panel, start, end, color, thickness, cv2.LINE_AA)

        event = projection.event or {}
        display_state = str(event.get("display_state", "airborne"))
        display_text_zh = str(
            event.get(
                "display_text_zh",
                self._fallback_chinese_status(projection),
            )
        )
        status_color = self._event_color(
            event.get("display_color", "fluorescent_green")
        )
        self.status_banner.draw(
            panel,
            text_zh=display_text_zh,
            fallback_text=display_state,
            color_bgr=status_color,
        )

        if projection.projection_valid and projection.ball_court_xy is not None:
            point = (
                float(projection.ball_court_xy[0]),
                float(projection.ball_court_xy[1]),
            )
            self.trail.append((point, projection.track_status))

        previous = None
        for point, status in self.trail:
            current = self._court_to_panel(
                point,
                scale,
                origin_x,
                origin_y,
            )
            if previous is not None:
                cv2.line(panel, previous, current, (0, 190, 255), 2, cv2.LINE_AA)
            previous = current

        if projection.projection_valid and projection.ball_court_xy is not None:
            raw_ball_point = self._court_to_panel(
                (
                    float(projection.ball_court_xy[0]),
                    float(projection.ball_court_xy[1]),
                ),
                scale,
                origin_x,
                origin_y,
            )
            ball_point, clipped_to_edge = self._visible_marker_point(
                raw_ball_point,
                width,
                frame_height,
            )
            color = status_color
            if clipped_to_edge:
                cv2.line(
                    panel,
                    (
                        int(
                            round(
                                ball_point[0]
                                - np.sign(raw_ball_point[0] - ball_point[0]) * 18
                            )
                        ),
                        int(
                            round(
                                ball_point[1]
                                - np.sign(raw_ball_point[1] - ball_point[1]) * 18
                            )
                        ),
                    ),
                    ball_point,
                    color,
                    3,
                    cv2.LINE_AA,
                )
            cv2.circle(panel, ball_point, 10, (0, 0, 0), -1, cv2.LINE_AA)
            if projection.predicted:
                cv2.circle(panel, ball_point, 7, color, 2, cv2.LINE_AA)
                cv2.circle(panel, ball_point, 2, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(panel, ball_point, 7, color, -1, cv2.LINE_AA)

            x_ft, y_ft = projection.ball_court_xy
            location_text = f"X={x_ft:.1f} Y={y_ft:.1f} FT"
            if projection.inside_court is False:
                location_text += " OUTSIDE COURT"
            if clipped_to_edge:
                location_text += " / EDGE MARKER"
            cv2.putText(
                panel,
                location_text,
                (12, frame_height - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

            event_names = event.get("events", [])
            event_text = str(event.get("phase", "airborne")).upper()
            if event_names:
                event_text += " / " + ",".join(event_names).upper()
            bounce_index = int(event.get("bounce_index_since_last_hit", 0))
            if bounce_index > 0:
                event_text += f" / BOUNCE={bounce_index}"
            cv2.putText(
                panel,
                event_text,
                (12, frame_height - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        status_text = (
            f"{projection.projection_status.upper()} "
            f"{projection.track_status or 'no_ball'}"
        )
        cv2.putText(
            panel,
            status_text,
            (12, frame_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _transform(
        self,
        panel_width: int,
        panel_height: int,
    ) -> tuple[float, float, float]:
        extended_width = self.layout.length_ft + 2.0 * self.outside_margin_ft
        extended_height = self.layout.width_ft + 2.0 * self.outside_margin_ft
        scale = min(
            (panel_width - 2.0 * self.margin_px) / extended_width,
            (panel_height - 2.0 * self.margin_px) / extended_height,
        )
        origin_x = (
            float(panel_width) - self.layout.length_ft * scale
        ) / 2.0
        origin_y = (
            float(panel_height) - self.layout.width_ft * scale
        ) / 2.0
        return scale, origin_x, origin_y

    @staticmethod
    def _court_to_panel(
        court_xy: tuple[float, float],
        scale: float,
        origin_x: float,
        origin_y: float,
    ) -> tuple[int, int]:
        x_ft, y_ft = court_xy
        return (
            int(round(origin_x + float(y_ft) * scale)),
            int(round(origin_y + float(x_ft) * scale)),
        )

    @staticmethod
    def _visible_marker_point(
        point: tuple[int, int],
        panel_width: int,
        panel_height: int,
        marker_padding: int = 12,
    ) -> tuple[tuple[int, int], bool]:
        x, y = point
        visible = (
            min(max(x, marker_padding), panel_width - marker_padding - 1),
            min(max(y, marker_padding), panel_height - marker_padding - 1),
        )
        return visible, visible != point

    @staticmethod
    def _event_color(name: str) -> tuple[int, int, int]:
        return {
            "fluorescent_green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "red": (0, 0, 255),
            "purple": (255, 0, 255),
        }.get(str(name), (0, 255, 0))

    @staticmethod
    def _fallback_chinese_status(projection: ProjectionResult) -> str:
        if projection.track_status is None:
            return "消失在屏幕"
        if not projection.projection_valid:
            return "投影不可用"
        if projection.predicted:
            return "短时预测"
        return "飞行"
