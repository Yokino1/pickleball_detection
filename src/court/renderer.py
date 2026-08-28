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
        outside_margin_ft: float = 6.0,
        trail_length: int = 15,
        status_font_path: str | None = None,
        status_font_size: int = 48,
        referee_result_hold_seconds: float = 5.0,
        court_event_hold_seconds: float = 2.0,
    ):
        self.layout = layout or CourtLayout()
        self.preferred_width = max(240, int(preferred_width))
        self.margin_px = max(8, int(margin_px))
        self.outside_margin_ft = max(0.0, float(outside_margin_ft))
        self.trail = deque(maxlen=max(1, int(trail_length)))
        self.referee_result_hold_seconds = max(
            0.0,
            float(referee_result_hold_seconds),
        )
        self.court_event_hold_seconds = max(
            0.0,
            float(court_event_hold_seconds),
        )
        self._latched_rally_result: dict | None = None
        self._latched_rally_result_until_s = float("-inf")
        self._latched_court_event: dict | None = None
        self._latched_court_event_until_s = float("-inf")
        self._score_history: list[dict] = []
        self._score_history_result_ids: set[str] = set()
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
        *,
        referee: dict | None = None,
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
        if referee is not None:
            visible_court_event = self._visible_court_event(event, referee)
            self._draw_referee(
                panel,
                referee,
                rally_result=self._visible_rally_result(referee),
            )
            self._draw_player_center_legend(panel)
            self._draw_score_history(panel)
            self._draw_court_event_feedback(panel, visible_court_event)
            self._draw_player_centers(
                panel,
                referee,
                scale=scale,
                origin_x=origin_x,
                origin_y=origin_y,
            )

        if projection.projection_valid and projection.ball_court_xy is not None:
            point = (
                float(projection.ball_court_xy[0]),
                float(projection.ball_court_xy[1]),
            )
            self.trail.append((point, projection.track_status))

        previous = None
        for point, _status in self.trail:
            current = self._court_to_panel(
                point,
                scale,
                origin_x,
                origin_y,
            )
            if previous is not None:
                clipped_segment = self._clip_line_to_panel(
                    previous,
                    current,
                    width,
                    frame_height,
                )
                if clipped_segment is not None:
                    cv2.line(
                        panel,
                        clipped_segment[0],
                        clipped_segment[1],
                        (0, 190, 255),
                        2,
                        cv2.LINE_AA,
                    )
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
                0.72,
                color,
                2,
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
                0.68,
                color,
                2,
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
            0.76,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return panel

    def _visible_court_event(
        self,
        event: dict,
        referee: dict,
    ) -> dict | None:
        """Hold short-lived bounce decisions long enough for video review."""
        timestamp_value = referee.get("timestamp_s")
        try:
            timestamp_s = float(timestamp_value)
        except (TypeError, ValueError):
            timestamp_s = float("nan")
        court_events = [
            str(name)
            for name in event.get("events", [])
            if str(name) in {
                "bounce_candidate",
                "out_of_bounds_bounce_candidate",
                "second_bounce_candidate",
            }
        ]
        notable_referee_events = [
            str(name)
            for name in referee.get("events", [])
            if str(name) in {
                "bounce_pending_uncertain",
                "first_bounce_recorded",
                "first_bounce_inside_unknown",
                "out_of_bounds",
                "second_bounce",
                "serve_net",
                "return_net",
                "serve_rejected_player_center_inside_court",
                "serve_rejected_wrong_service_owner",
            }
        ]
        if court_events:
            self._latched_court_event = {
                "court_events": court_events,
                "referee_events": notable_referee_events,
                "referee_phase": str(referee.get("phase", "unknown")),
                "contact_frame_index": event.get("contact_frame_index"),
            }
            self._latched_court_event_until_s = (
                timestamp_s + self.court_event_hold_seconds
                if np.isfinite(timestamp_s)
                else float("-inf")
            )
            return dict(self._latched_court_event)

        if notable_referee_events:
            if (
                self._latched_court_event is not None
                and np.isfinite(timestamp_s)
                and timestamp_s <= self._latched_court_event_until_s
            ):
                merged = list(
                    dict.fromkeys(
                        (
                            *self._latched_court_event.get(
                                "referee_events",
                                [],
                            ),
                            *notable_referee_events,
                        )
                    )
                )
                self._latched_court_event["referee_events"] = merged
                self._latched_court_event["referee_phase"] = str(
                    referee.get("phase", "unknown")
                )
            else:
                self._latched_court_event = {
                    "court_events": [],
                    "referee_events": notable_referee_events,
                    "referee_phase": str(referee.get("phase", "unknown")),
                    "contact_frame_index": None,
                }
            self._latched_court_event_until_s = (
                timestamp_s + self.court_event_hold_seconds
                if np.isfinite(timestamp_s)
                else float("-inf")
            )
            return dict(self._latched_court_event)

        if self._latched_court_event is None or not np.isfinite(timestamp_s):
            return None
        if timestamp_s > self._latched_court_event_until_s:
            self._latched_court_event = None
            return None
        return dict(self._latched_court_event)

    def _visible_rally_result(self, referee: dict) -> dict | None:
        """Latch a one-shot terminal result for human-readable video output."""
        current = referee.get("rally_result")
        timestamp_value = referee.get("timestamp_s")
        try:
            timestamp_s = float(timestamp_value)
        except (TypeError, ValueError):
            timestamp_s = float("nan")
        if isinstance(current, dict):
            self._record_score_history(current)
            self._latched_rally_result = dict(current)
            self._latched_rally_result_until_s = (
                timestamp_s + self.referee_result_hold_seconds
                if np.isfinite(timestamp_s)
                else float("-inf")
            )
            return current

        latched = self._latched_rally_result
        if latched is None or not np.isfinite(timestamp_s):
            return None
        if timestamp_s > self._latched_rally_result_until_s:
            self._latched_rally_result = None
            return None
        current_rally_id = referee.get("rally_id")
        latched_rally_id = latched.get("rally_id")
        if (
            current_rally_id is not None
            and latched_rally_id is not None
            and current_rally_id != latched_rally_id
        ):
            self._latched_rally_result = None
            return None
        return latched

    def _record_score_history(self, rally_result: dict) -> None:
        """Persist each awarded point once for the right-panel score log."""
        if rally_result.get("point_awarded_to") is None:
            return
        identity_value = rally_result.get("result_id")
        if identity_value is None:
            identity_value = rally_result.get("rally_id")
        identity = str(identity_value)
        if identity in self._score_history_result_ids:
            return
        score_after = rally_result.get("score_after")
        if not isinstance(score_after, dict):
            return
        self._score_history_result_ids.add(identity)
        self._score_history.append(dict(rally_result))

    def _score_history_lines(self) -> list[tuple[str, str]]:
        """Return Chinese display text plus an ASCII fallback for every point."""
        side_zh = {"left": "左侧", "right": "右侧"}
        reason_zh = {
            "serve_net": "下网",
            "return_net": "下网",
            "out_of_bounds": "出界",
            "second_bounce": "二弹",
        }
        lines: list[tuple[str, str]] = []
        for index, result in enumerate(self._score_history, start=1):
            score = result.get("score_after", {})
            left_score = int(score.get("left", 0))
            right_score = int(score.get("right", 0))
            fault = str(result.get("fault_side", ""))
            terminal = str(result.get("terminal_event", "point"))
            chinese_reason = reason_zh.get(terminal, "得分")
            chinese_side = side_zh.get(fault, fault.upper() or "未知侧")
            text_zh = (
                f"{index}.  {left_score}:{right_score}   "
                f"{chinese_side}{chinese_reason}"
            )
            fallback = (
                f"{index}.  {left_score}:{right_score}   "
                f"{fault.upper() or 'UNKNOWN'} "
                f"{terminal.replace('_', ' ').upper()}"
            )
            lines.append((text_zh, fallback))
        return lines

    def _draw_score_history(self, panel: np.ndarray) -> None:
        """Draw the cumulative point-by-point score reason ledger."""
        width = int(panel.shape[1])
        top = 500
        bottom = min(int(panel.shape[0]) - 720, 1180)
        if bottom <= top + 80:
            return
        cv2.rectangle(
            panel,
            (16, top),
            (max(16, width - 16), bottom),
            (12, 12, 12),
            cv2.FILLED,
        )
        cv2.rectangle(
            panel,
            (16, top),
            (max(16, width - 16), bottom),
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )
        self.status_banner.draw_text(
            panel,
            text_zh="得分记录（左 : 右）",
            fallback_text="SCORE HISTORY (LEFT : RIGHT)",
            origin=(34, top + 22),
            color_bgr=(0, 220, 255),
            font_size=52,
        )
        lines = self._score_history_lines()
        if not lines:
            self.status_banner.draw_text(
                panel,
                text_zh="暂无得分",
                fallback_text="NO POINTS YET",
                origin=(40, top + 100),
                color_bgr=(210, 210, 210),
                font_size=44,
            )
            return
        available_height = max(1, bottom - (top + 104) - 16)
        line_height = min(68, max(40, available_height // len(lines)))
        font_size = min(52, max(32, line_height - 12))
        visible_count = max(1, available_height // line_height)
        visible_lines = lines[-visible_count:]
        for line_index, (text_zh, fallback) in enumerate(visible_lines):
            self.status_banner.draw_text(
                panel,
                text_zh=text_zh,
                fallback_text=fallback,
                origin=(42, top + 104 + line_index * line_height),
                color_bgr=(235, 235, 235),
                font_size=font_size,
            )

    @staticmethod
    def _draw_referee(
        panel: np.ndarray,
        referee: dict,
        *,
        rally_result: dict | None = None,
    ) -> None:
        """Draw the downstream demo state without affecting adjudication."""
        score = referee.get("score", {})
        if not isinstance(score, dict):
            score = {}
        score_items = list(score.items())[:2]
        if len(score_items) == 2:
            (first_name, first_score), (second_name, second_score) = score_items
            score_text = (
                f"SCORE  {str(first_name).upper()} {first_score} : "
                f"{second_score} {str(second_name).upper()}"
            )
        else:
            score_text = "SCORE  -- : --"

        state = str(referee.get("phase", "unknown")).upper()
        server = referee.get("server_side")
        service_owner = referee.get("service_owner_side")
        state_text = state
        service_text = "SERVER=--  SERVICE OWNER=--"
        if server is not None or service_owner is not None:
            service_text = (
                "SERVER="
                f"{str(server).upper() if server is not None else '--'}  "
                "SERVICE OWNER="
                f"{str(service_owner).upper() if service_owner is not None else '--'}"
            )
        player_outside = bool(
            referee.get("eligible_player_center_outside_court", False)
        )
        player_text = (
            "ELIGIBLE PLAYER CENTER: OUTSIDE COURT"
            if player_outside
            else "ELIGIBLE PLAYER CENTER: INSIDE / UNAVAILABLE"
        )
        color = (0, 255, 0)
        if isinstance(rally_result, dict):
            terminal = str(
                rally_result.get("terminal_event", "rally_ended")
            ).replace("_", " ").upper()
            point_side = rally_result.get("point_awarded_to")
            next_server = rally_result.get("next_server_side")
            fault_side = rally_result.get("fault_side")
            state_text = (
                f"POINT={str(point_side).upper()}  REASON={terminal}"
                if point_side is not None
                else f"NO POINT  REASON={terminal}"
            )
            service_text = (
                "NEXT SERVE="
                f"{str(next_server).upper() if next_server is not None else '--'}"
            )
            if fault_side is not None:
                service_text += f"  FAULT={str(fault_side).upper()}"
            color = (0, 220, 255)

        width = int(panel.shape[1])
        score_left = 32
        score_target_width = max(1, int(width * 0.9))
        score_unit_width = max(
            1,
            cv2.getTextSize(
                score_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                6,
            )[0][0],
        )
        score_scale = score_target_width / float(score_unit_width)
        score_size = cv2.getTextSize(
            score_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            score_scale,
            6,
        )[0]
        score_left = max(20, (width - score_size[0]) // 2)
        cv2.rectangle(
            panel,
            (16, 104),
            (max(16, width - 16), 438),
            (12, 12, 12),
            cv2.FILLED,
        )
        cv2.rectangle(
            panel,
            (16, 104),
            (max(16, width - 16), 438),
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            score_text,
            (score_left, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            score_scale,
            color,
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            state_text,
            (34, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.08,
            (220, 220, 220),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            service_text,
            (34, 354),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.98,
            (255, 220, 120),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            player_text,
            (34, 410),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 255, 0) if player_outside else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_player_center_legend(panel: np.ndarray) -> None:
        """Make the projected player-center crosses self-explanatory."""
        cv2.drawMarker(
            panel,
            (38, 470),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            panel,
            (76, 470),
            (255, 128, 0),
            cv2.MARKER_CROSS,
            24,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "PLAYER BBOX CENTERS: YELLOW=IN / BLUE=OUT",
            (110, 480),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_court_event_feedback(
        panel: np.ndarray,
        event_feedback: dict | None,
    ) -> None:
        if not isinstance(event_feedback, dict):
            return
        court_events = set(event_feedback.get("court_events", []))
        referee_events = set(event_feedback.get("referee_events", []))
        phase = str(event_feedback.get("referee_phase", "unknown")).upper()

        if "second_bounce_candidate" in court_events:
            title = "COURT EVENT: SECOND BOUNCE CANDIDATE"
        elif "out_of_bounds_bounce_candidate" in court_events:
            title = "COURT EVENT: OUT-OF-BOUNDS BOUNCE CANDIDATE"
        elif "bounce_candidate" in court_events:
            title = "COURT EVENT: BOUNCE CANDIDATE"
        elif "serve_rejected_player_center_inside_court" in referee_events:
            title = "REFEREE: SERVE REJECTED"
        elif "serve_rejected_wrong_service_owner" in referee_events:
            title = "REFEREE: WRONG SERVER REJECTED"
        else:
            title = "REFEREE EVENT"

        if "second_bounce" in referee_events:
            decision = "SCORED: SECOND BOUNCE"
        elif "out_of_bounds" in referee_events:
            decision = "SCORED: OUT OF BOUNDS"
        elif "serve_net" in referee_events:
            decision = "SCORED: SERVE NET"
        elif "return_net" in referee_events:
            decision = "SCORED: RETURN NET"
        elif "first_bounce_recorded" in referee_events:
            decision = "REFEREE: FIRST BOUNCE RECORDED"
        elif "bounce_pending_uncertain" in referee_events:
            decision = "REFEREE: BOUNCE PENDING / NO SCORE YET"
        elif "serve_rejected_player_center_inside_court" in referee_events:
            decision = "NO SCORE: PLAYER CENTER INSIDE COURT"
        elif "serve_rejected_wrong_service_owner" in referee_events:
            decision = "NO SCORE: SIDE DOES NOT OWN SERVE"
        elif phase != "IN_RALLY":
            decision = f"NO SCORE: NOT IN ACTIVE RALLY ({phase})"
        else:
            decision = "REFEREE: CANDIDATE UNDER REVIEW"

        frame_value = event_feedback.get("contact_frame_index")
        frame_text = (
            f"  CONTACT FRAME={int(frame_value)}"
            if frame_value is not None
            else ""
        )
        color = (0, 220, 255) if decision.startswith("SCORED") else (0, 255, 255)
        width = int(panel.shape[1])
        cv2.rectangle(
            panel,
            (16, 1210),
            (max(16, width - 16), 1400),
            (12, 12, 12),
            cv2.FILLED,
        )
        cv2.rectangle(
            panel,
            (16, 1210),
            (max(16, width - 16), 1400),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            title,
            (32, 1280),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.18,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            decision + frame_text,
            (32, 1355),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.98,
            (230, 230, 230),
            3,
            cv2.LINE_AA,
        )

    def _draw_player_centers(
        self,
        panel: np.ndarray,
        referee: dict,
        *,
        scale: float,
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Show the projected player-box centers used by serve validation."""
        height, width = panel.shape[:2]
        for index, value in enumerate(
            referee.get("eligible_player_centers_court_xy", []),
            start=1,
        ):
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            try:
                court_xy = (float(value[0]), float(value[1]))
            except (TypeError, ValueError, OverflowError):
                continue
            raw_point = self._court_to_panel(
                court_xy,
                scale,
                origin_x,
                origin_y,
            )
            point, clipped = self._visible_marker_point(
                raw_point,
                width,
                height,
                marker_padding=18,
            )
            color = (
                (0, 255, 255)
                if self.layout.contains(court_xy)
                else (255, 128, 0)
            )
            cv2.drawMarker(
                panel,
                point,
                color,
                cv2.MARKER_CROSS,
                22,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"P{index}",
                (point[0] + 12, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            if clipped:
                cv2.circle(panel, point, 13, color, 2, cv2.LINE_AA)

    def _transform(
        self,
        panel_width: int,
        panel_height: int,
    ) -> tuple[float, float, float]:
        extended_width = self.layout.length_ft + 2.0 * self.outside_margin_ft
        extended_height = self.layout.width_ft + 2.0 * self.outside_margin_ft
        reserved_top = min(1300.0, float(panel_height) * 0.58)
        available_height = max(
            1.0,
            float(panel_height) - reserved_top - self.margin_px,
        )
        scale = min(
            (panel_width - 2.0 * self.margin_px) / extended_width,
            available_height / extended_height,
        )
        origin_x = (
            float(panel_width) - self.layout.length_ft * scale
        ) / 2.0
        origin_y = reserved_top + (
            available_height - self.layout.width_ft * scale
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
    def _clip_line_to_panel(
        start: tuple[int, int],
        end: tuple[int, int],
        panel_width: int,
        panel_height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Clip arbitrary Python-sized coordinates before calling OpenCV."""
        coordinate_limit = 1 << 30

        def bounded(point: tuple[int, int]) -> tuple[int, int]:
            return (
                min(max(int(point[0]), -coordinate_limit), coordinate_limit),
                min(max(int(point[1]), -coordinate_limit), coordinate_limit),
            )

        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, int(panel_width), int(panel_height)),
            bounded(start),
            bounded(end),
        )
        if not visible:
            return None
        return clipped_start, clipped_end

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
