"""
Side-view pickleball video analysis — video in, annotated video + JSONL out.

Usage::

    # Minimal — uses config defaults
    python apps/run_sideview_video.py

    # Override everything
    python apps/run_sideview_video.py `
        --config configs/sideview_config.yaml `
        --input data/samples/test.mp4 `
        --output outputs/sideview_overlay.mp4 `
        --jsonl outputs/sideview.jsonl `
        --preview `
        --trail 3

When ``models.ball_model`` is ``null`` the app runs with a
``NullBallDetector`` (always returns no detection).  This is
intentional — it lets you validate the full pipeline, JSONL records,
and overlay rendering before a real model is available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Bootstrap: add project root so we can import src/*
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tracking.ball_detector import BallDetector, NullBallDetector, YoloBallDetector
from src.tracking.ball_track import SimpleBallTracker
from src.tracking.events import EventDetector
from src.tracking.pipeline import SideViewPipeline
from src.tracking.player_detector import NullPlayerDetector, PlayerDetector, YoloPlayerDetector
from src.tracking.types import BallTrack, FrameResult

# Optional court module
try:
    from src.court.layout import CourtLayout
    from src.court.observation import CourtObservation
    from src.court.projector import CourtProjector
    from src.court.renderer import CourtRenderer
    COURT_OK = True
except ImportError as e:
    COURT_OK = False
    CourtLayout = None
    CourtObservation = None
    CourtProjector = None
    CourtRenderer = None

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------
C_WHITE = (255, 255, 255)
C_GREEN = (0, 255, 0)
C_YELLOW = (0, 255, 255)
C_ORANGE = (0, 165, 255)
C_RED = (0, 0, 255)
C_CYAN = (255, 255, 0)
C_GREY = (128, 128, 128)
C_DARK = (40, 40, 40)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_detector(cfg: dict) -> BallDetector:
    model_path = cfg.get("models", {}).get("ball_model", None)
    if model_path is None or model_path == "null":
        print("[detector] ball_model is null — using NullBallDetector")
        return NullBallDetector()

    print(f"[detector] loading ball model: {model_path}")
    try:
        raw_conf = cfg.get("tracking", {}).get("min_conf", 0.25)
        imgsz = cfg.get("models", {}).get("imgsz", 640)
        exclude = cfg.get("models", {}).get("exclude_region", None)
        if exclude is not None:
            print(f"[detector] exclusion zone: {exclude}")
        return YoloBallDetector(
            model_path=model_path,
            ball_class_id=cfg.get("models", {}).get("ball_class_id", 0),
            conf_threshold=raw_conf,
            imgsz=imgsz,
            exclude_region=exclude,
        )
    except FileNotFoundError as exc:
        print(f"[detector] ERROR: {exc}")
        print("[detector] falling back to NullBallDetector")
        return NullBallDetector()


def build_tracker(cfg: dict) -> SimpleBallTracker:
    t = cfg.get("tracking", {})
    return SimpleBallTracker(
        min_conf=t.get("min_conf", 0.25),
        max_jump_px=t.get("max_jump_px", 180),
        max_predict_frames=t.get("max_predict_frames", 3),
        smoothing_window=t.get("smoothing_window", 3),
    )


def build_events(cfg: dict) -> EventDetector:
    return EventDetector()


def build_player_detector(cfg: dict) -> PlayerDetector:
    if not cfg.get("players", {}).get("enabled", False):
        return NullPlayerDetector()

    model_path = cfg.get("models", {}).get("player_model", None)
    if model_path is None or model_path == "null":
        return NullPlayerDetector()

    print(f"[players] loading model: {model_path}")
    try:
        return YoloPlayerDetector(model_path=model_path)
    except Exception as exc:
        print(f"[players] ERROR: {exc}, falling back to NullPlayerDetector")
        return NullPlayerDetector()


# ---------------------------------------------------------------------------
# Overlay renderer
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """Draws ball trail, status bar, mini-court, and optional court overlay."""

    def __init__(
        self,
        trail_length: int = 20,
        court_renderer: Optional[CourtRenderer] = None,
        court_layout: Optional[CourtLayout] = None,
        court_projector: Optional[CourtProjector] = None,
        draw_image_court: bool = True,
        draw_mini_court: bool = True,
        static_court_observation: Optional[CourtObservation] = None,
    ):
        self.trail: deque[tuple[float, float]] = deque(maxlen=trail_length)
        self._fps_history: deque[float] = deque(maxlen=30)
        self._last_t = time.perf_counter()
        self.court_renderer = court_renderer
        self.court_layout = court_layout
        self.court_projector = court_projector
        self.draw_image_court = draw_image_court
        self.draw_mini_court = draw_mini_court
        self.static_court_observation = static_court_observation
        self._last_proj_quality: dict = {}

    def draw(
        self,
        frame: np.ndarray,
        result: FrameResult,
        detector_label: str,
    ) -> np.ndarray:
        """Return *frame* with overlay drawn in-place."""
        h, w = frame.shape[:2]
        track = result.ball_track

        # Cache projection quality for per-point rendering
        self._last_proj_quality = result.diagnostics.get("projection_quality", {})

        # --- ball trail --------------------------------------------------
        if track and track.center is not None and track.status != "absent":
            self.trail.append((track.center[0], track.center[1]))

        self._draw_trail(frame)

        # --- ball marker -------------------------------------------------
        if track and track.center is not None:
            self._draw_ball_marker(frame, track)

        # --- tracked court keypoints (from optical flow) -----------------
        kp_obs = result.diagnostics.get("kp_observation")
        if kp_obs and kp_obs.get("num_visible", 0) > 0:
            self._draw_tracked_keypoints(frame, kp_obs)
        elif (
            self.static_court_observation is not None
            and self.court_renderer is not None
        ):
            self.court_renderer.draw_image_keypoints(
                frame, self.static_court_observation
            )

        # --- image-space court lines (only when homography available) -----
        if (
            self.draw_image_court
            and self.court_renderer is not None
            and self.court_projector is not None
            and self.court_projector.is_available
            and self.court_layout is not None
        ):
            self.court_renderer.draw_image_court_lines(
                frame, self.court_projector, self.court_layout
            )

        # --- mini-court (always, if available) ---------------------------
        if (
            self.draw_mini_court
            and self.court_renderer is not None
            and self.court_layout is not None
        ):
            ball_ft = None
            if result.court and result.court.ball_court_xy:
                ball_ft = tuple(result.court.ball_court_xy)
            proj_status = result.court.projection_status if result.court else "none"
            self.court_renderer.draw_mini_court(
                frame, self.court_layout,
                ball_court_xy=ball_ft,
                projection_status=proj_status,
            )

        # --- status bar --------------------------------------------------
        self._draw_status_bar(frame, result, detector_label, w)

        # --- events ------------------------------------------------------
        self._draw_events(frame, result.events, h)

        return frame

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_trail(self, frame: np.ndarray) -> None:
        if len(self.trail) < 2:
            return
        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = (i + 1) / len(pts)
            colour = (
                int(0 * alpha),
                int(140 + 115 * alpha),
                int(200 + 55 * alpha),
            )
            cv2.line(
                frame,
                (int(pts[i - 1][0]), int(pts[i - 1][1])),
                (int(pts[i][0]), int(pts[i][1])),
                colour, max(1, int(2 * alpha)), cv2.LINE_AA,
            )

    def _draw_roi(self, frame: np.ndarray, track: BallTrack) -> None:
        if track.roi is None:
            return
        x1, y1, x2, y2 = [int(v) for v in track.roi]
        self._dashed_rect(frame, x1, y1, x2, y2, C_GREEN, thickness=1, dash=12)

    def _draw_ball_marker(self, frame: np.ndarray, track: BallTrack) -> None:
        if track.center is None:
            return
        cx, cy = int(track.center[0]), int(track.center[1])

        if track.status == "observed":
            colour = C_GREEN
            cv2.circle(frame, (cx, cy), self._ball_radius(track), colour, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(2, self._ball_radius(track) // 4), colour, -1, cv2.LINE_AA)
            cv2.putText(
                frame, f"{track.confidence:.0%}", (cx + self._ball_radius(track) + 6, cy - self._ball_radius(track) - 4),
                FONT, 0.45, C_GREEN, 1, cv2.LINE_AA,
            )
        elif track.status == "predicted":
            colour = C_ORANGE
            cv2.circle(frame, (cx, cy), self._ball_radius(track), colour, 1, cv2.LINE_AA)
            cv2.putText(
                frame, f"pred {track.missing_frames}f", (cx + self._ball_radius(track) + 6, cy - self._ball_radius(track) - 4),
                FONT, 0.4, C_ORANGE, 1, cv2.LINE_AA,
            )
        # absent → no marker drawn

    @staticmethod
    def _ball_radius(track: BallTrack) -> int:
        if track.bbox is not None and len(track.bbox) >= 4:
            x1, y1, x2, y2 = track.bbox[:4]
            w = max(1.0, float(x2) - float(x1))
            h = max(1.0, float(y2) - float(y1))
            return max(4, min(12, int(round(max(w, h) * 0.5))))
        return 6

    def _draw_events(self, frame: np.ndarray, events: list[str], h: int) -> None:
        y0 = h - 14
        for ev in reversed(events[-4:]):  # max 4 events to avoid clutter
            if "bounce" in ev:
                colour = C_CYAN
            elif "lost" in ev or "missing" in ev:
                colour = C_RED
            elif "unavailable" in ev:
                colour = C_GREY
            else:
                colour = C_WHITE
            (tw, th), _ = cv2.getTextSize(ev, FONT, 0.45, 1)
            cv2.putText(frame, ev, (8, y0), FONT, 0.45, colour, 1, cv2.LINE_AA)
            y0 -= th + 5

    def _draw_status_bar(
        self, frame: np.ndarray, result: FrameResult, detector_label: str, w: int
    ) -> None:
        # Semi-transparent bar
        bar = frame[0:30, 0:w].copy()
        cv2.rectangle(frame, (0, 0), (w, 30), C_DARK, -1)
        frame[0:30, 0:w] = cv2.addWeighted(bar, 0.4, frame[0:30, 0:w], 0.6, 0)

        # FPS
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        fps = np.mean(self._fps_history) if self._fps_history else 0.0

        # Build status line
        track = result.ball_track
        if track:
            state_str = track.status.upper()
            conf_str = f"{track.confidence:.0%}"
        else:
            state_str = "INIT"
            conf_str = "-"

        proj_status = result.court.projection_status if result.court else "none"

        # Keypoint tracking info
        kp_info = ""
        kp_quality = result.diagnostics.get("kp_tracker_quality")
        if kp_quality:
            ms = kp_quality.get("motion_state", kp_quality.get("tracker", "?"))
            active = kp_quality.get("active_points", "?")
            total = kp_quality.get("initial_points", kp_quality.get("annotated_templates", "?"))
            kp_info = f"  kp:{active}/{total} {ms}"

        # Projection quality summary
        proj_q = result.diagnostics.get("projection_quality", {})
        proj_detail = ""
        if proj_q.get("per_point"):
            n_pts = len(proj_q["per_point"])
            n_inlier = sum(1 for pp in proj_q["per_point"] if pp.get("is_inlier"))
            mean_err = proj_q.get("mean_reproj_error_px", 999)
            cache_age = proj_q.get("cache_age", 0)
            proj_detail = f"  H:{n_inlier}/{n_pts}i err={mean_err:.1f}px"
            if cache_age > 0:
                proj_detail += f" cache={cache_age}f"

        line = (
            f"frame:{result.frame_index:06d}  "
            f"fps:{fps:5.1f}  "
            f"det:{detector_label}  "
            f"ball:{state_str}  "
            f"conf:{conf_str}  "
            f"proj:{proj_status}"
            f"{kp_info}"
            f"{proj_detail}"
        )
        cv2.putText(frame, line, (8, 20), FONT, 0.4, C_WHITE, 1, cv2.LINE_AA)

        if detector_label == "null":
            cv2.putText(
                frame, "NO DETECTOR", (w - 140, 20),
                FONT, 0.45, C_RED, 1, cv2.LINE_AA,
            )

    def _draw_tracked_keypoints(self, frame: np.ndarray, kp_obs: dict) -> None:
        """Draw tracked court keypoints with reprojection error annotation."""
        kps = kp_obs.get("keypoints", {})
        if not kps:
            return
        from src.court.layout import KEYPOINT_NAMES

        # Try to get per-point reprojection errors from diagnostics
        proj_quality = self._last_proj_quality or {}
        per_point_map = {}
        for pp in proj_quality.get("per_point", []):
            per_point_map[pp["kp_idx"]] = pp

        for kp_str, (px, py) in kps.items():
            kp_idx = int(kp_str)
            name = KEYPOINT_NAMES.get(kp_idx, kp_str)

            # Reprojection error colour
            pp = per_point_map.get(kp_idx, {})
            reproj = pp.get("reproj_error_px", None)
            is_inlier = pp.get("is_inlier", True)

            if reproj is not None:
                if reproj < 3.0:
                    err_col = (0, 255, 0)    # 绿 — 好
                elif reproj < 8.0:
                    err_col = (0, 255, 255)  # 黄 — 可疑
                else:
                    err_col = (0, 0, 255)    # 红 — 差
            else:
                err_col = (128, 128, 128)    # 灰 — 无数据

            # RANSAC outlier: red circle border
            if not is_inlier:
                cv2.circle(frame, (int(px), int(py)), 8, (0, 0, 255), 2, cv2.LINE_AA)

            # Point marker
            cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 255), -1, cv2.LINE_AA)

            # Index label
            cv2.putText(frame, str(kp_idx),
                       (int(px) + 6, int(py) - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

            # Reprojection error label
            if reproj is not None:
                label = f"{reproj:.1f}"
            else:
                label = "?"
            cv2.putText(frame, label,
                       (int(px) + 6, int(py) + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, err_col, 1, cv2.LINE_AA)

    @staticmethod
    def _dashed_rect(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        colour: tuple[int, int, int],
        thickness: int = 1,
        dash: int = 10,
    ) -> None:
        for x in range(x1, x2, dash * 2):
            xe = min(x + dash, x2)
            cv2.line(frame, (x, y1), (xe, y1), colour, thickness)
            cv2.line(frame, (x, y2), (xe, y2), colour, thickness)
        for y in range(y1, y2, dash * 2):
            ye = min(y + dash, y2)
            cv2.line(frame, (x1, y), (x1, ye), colour, thickness)
            cv2.line(frame, (x2, y), (x2, ye), colour, thickness)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Side-view pickleball video analysis")
    parser.add_argument("--config", default="configs/sideview_config.yaml",
                        help="Path to YAML config")
    parser.add_argument("--input", default=None,
                        help="Video file (overrides config camera.source)")
    parser.add_argument("--output", default=None,
                        help="Output video path")
    parser.add_argument("--jsonl", default=None,
                        help="JSONL output path")
    parser.add_argument("--preview", action="store_true", default=False,
                        help="Show live preview window")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="Disable live preview window (default)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Start processing from frame N (default: 0 or calib ref frame)")
    parser.add_argument("--trail", type=int, default=20,
                        help="Trail length in frames")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Compare basic clamp vs motion-aware clamp side-by-side")
    parser.add_argument("--auto-calibrate", action="store_true", default=False,
                        help="Auto-detect court keypoints via CV (no manual calibration needed)")
    parser.add_argument("--calib-frame", type=int, default=None,
                        help="Reference frame for auto court detection (overrides config)")
    parser.add_argument("--detect-every", type=int, default=0,
                        help="Re-run CV court detection every N frames (0=off)")
    parser.add_argument("--tracker", type=str, default=None,
                        help="Court tracker: pose | template | lk (overrides config)")
    parser.add_argument("--display-scale", type=float, default=0.5,
                        help="Preview window scale factor (default: 0.5, fits 4K→1080p)")

    args = parser.parse_args()

    # --- config ----------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    cfg = load_config(str(config_path))

    # --- input source ----------------------------------------------------
    video_path = args.input or cfg.get("camera", {}).get("source", "data/samples/test.mp4")
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[input] {video_path}  ({width}x{height}, {total_frames} frames, {fps_in:.1f} fps)")

    # --- output ----------------------------------------------------------
    out_jsonl = args.jsonl
    if out_jsonl is None and cfg.get("output", {}).get("save_jsonl", True):
        out_jsonl = "outputs/sideview.jsonl"
    jsonl_fh = None
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = open(out_jsonl, "w", encoding="utf-8")
        print(f"[output] JSONL → {out_jsonl}")

    out_video = args.output
    if out_video is None and cfg.get("output", {}).get("save_video", False):
        out_video = "outputs/sideview_overlay.mp4"
    vw = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_video), fourcc, fps_in if fps_in > 0 else 30.0,
                             (width, height))
        print(f"[output] video → {out_video}")

    # --- pipeline components ---------------------------------------------
    detector = build_detector(cfg)
    tracker = build_tracker(cfg)
    event_detector = build_events(cfg)
    player_detector = build_player_detector(cfg)

    # Court (always create layout; projector if calibration available)
    court_layout = None
    court_projector = None
    court_observation = None
    keypoint_tracker = None
    keypoint_tracker_a = None
    court_projector_a = None
    court_renderer = None
    kp_tracking_mode = "none"

    if cfg.get("court", {}).get("enabled", True) and COURT_OK:
        court_layout = CourtLayout(
            view=cfg.get("court", {}).get("overlay_view", "sideview")
        )
        court_renderer = CourtRenderer()

        tracker_type = args.tracker or cfg.get("court", {}).get("tracker", "template")
        calib_ref_frame = 0
        kp_path = cfg.get("court", {}).get("keypoints_path", None)

        # ── CLI override ──
        if args.auto_calibrate:
            kp_path = "auto"

        # ── obtain court observation (auto-CV or manual YAML) ──
        if kp_path == "auto" or kp_path is None:
            from tools.detect_court_cv import detect_court_single_frame

            calib_ref_frame = (
                args.calib_frame
                if args.calib_frame is not None
                else cfg.get("court", {}).get("calibration_frame", 0)
            )
            print(f"[court] auto-detecting keypoints on frame {calib_ref_frame}...")

            cap.set(cv2.CAP_PROP_POS_FRAMES, calib_ref_frame)
            ok_ref, ref_frame = cap.read()
            if ok_ref:
                kps_dict = detect_court_single_frame(ref_frame, downscale=2)
                if kps_dict is not None and len(kps_dict) >= 4:
                    court_observation = CourtObservation.from_calibration(kps_dict)
                    print(f"[court] auto-detected {court_observation.num_visible} keypoints "
                          f"on frame {calib_ref_frame}")
                else:
                    print(f"[court] auto-detection failed on frame {calib_ref_frame} "
                          f"— court tracking disabled")
                    court_observation = None
            else:
                print(f"[court] WARNING: cannot read frame {calib_ref_frame} for auto-detection")
                court_observation = None
        elif kp_path:
            from src.court.calibration_compat import load_court_observation

            # Load the annotated keypoints
            court_observation = load_court_observation(kp_path)

            # Read reference_frame from the YAML
            with open(kp_path, "r", encoding="utf-8") as _f:
                _calib_data = yaml.safe_load(_f)
                calib_ref_frame = _calib_data.get("calibration", {}).get("reference_frame", 0)

            if court_observation is not None and court_observation.is_reliable:
                print(f"[court] loaded {court_observation.num_visible} keypoints "
                      f"(annotated on frame {calib_ref_frame})")
            elif court_observation is not None:
                print(f"[court] only {court_observation.num_visible} keypoints — "
                      f"need ≥4 for projection, using static mode")

        # ── initialize tracker (shared path for auto and manual) ──
        if args.detect_every > 0:
            # Per-frame detection mode — keep initial observation for first frame,
            # then re-detect each frame in the main loop. No tracker needed.
            print(f"[court] per-frame calibration mode (no tracking)")
            kp_tracking_mode = "perframe"
        if (court_observation is not None and court_observation.is_reliable
                and kp_tracking_mode != "perframe"):
            # Seek to the calibration reference frame for tracker init
            cap.set(cv2.CAP_PROP_POS_FRAMES, calib_ref_frame)
            ok_ref, ref_frame = cap.read()
            if ok_ref:
                tpl_cfg = cfg.get("court", {})
                if tracker_type == "pose":
                    from src.court.pose_tracker import CourtPoseTracker

                    keypoint_tracker = CourtPoseTracker(
                        template_half_size=tpl_cfg.get("template_half_size", 15),
                        search_radius=tpl_cfg.get("search_radius", 50),
                        recovery_search_radius=tpl_cfg.get("recovery_search_radius", 120),
                        min_ncc_score=tpl_cfg.get("min_ncc_score", 0.60),
                        ransac_reproj_px=tpl_cfg.get("ransac_reproj_px", 5.0),
                        line_band_px=tpl_cfg.get("line_band_px", 4),
                        line_score_min=tpl_cfg.get("line_score_min", 0.30),
                        max_hold_frames=tpl_cfg.get("max_hold_frames", 15),
                        motion_bridge_enabled=tpl_cfg.get("motion_bridge_enabled", True),
                        line_refiner_enabled=tpl_cfg.get("line_refiner_enabled", True),
                        anchor_mode=tpl_cfg.get("anchor_mode", False),
                        h_smooth_alpha=tpl_cfg.get("h_smooth_alpha", 0.55),
                    )
                    keypoint_tracker.init(ref_frame, dict(court_observation.keypoints))
                    kp_tracking_mode = "pose"
                    print(f"[court] pose tracker initialised on frame {calib_ref_frame} "
                          f"with {keypoint_tracker.active_count} keypoints "
                          f"(search_radius={keypoint_tracker._search_radius}, "
                          f"ransac_reproj={keypoint_tracker._ransac_reproj}px)")

                    # Compare mode: baseline template tracker
                    keypoint_tracker_a = None
                    court_projector_a = None
                    if args.compare:
                        from src.court.template_tracker import TemplateKeypointTracker
                        keypoint_tracker_a = TemplateKeypointTracker(
                            template_half_size=tpl_cfg.get("template_half_size", 15),
                            search_radius=tpl_cfg.get("compare_search_radius", 50),
                            min_ncc_score=tpl_cfg.get("compare_min_ncc_score",
                                                      tpl_cfg.get("min_ncc_score", 0.60)),
                            recovery_radius=tpl_cfg.get("recovery_radius", 120),
                            max_lost_frames=tpl_cfg.get("max_lost_frames", 5),
                            smooth_alpha=tpl_cfg.get("compare_smooth_alpha",
                                                     tpl_cfg.get("smooth_alpha", 0.50)),
                            max_jump_px=tpl_cfg.get("compare_max_jump_px",
                                                    tpl_cfg.get("max_jump_px", 45)),
                            geometry_consistency_enabled=tpl_cfg.get("geometry_consistency_enabled", True),
                            geometry_ransac_px=tpl_cfg.get("geometry_ransac_px", 6.0),
                            geometry_snap_px=tpl_cfg.get("geometry_snap_px", 12.0),
                        )
                        keypoint_tracker_a.init(ref_frame, dict(court_observation.keypoints))
                        court_projector_a = CourtProjector(
                            max_reproj_error_px=tpl_cfg.get("max_reproj_error_px", 15.0),
                            min_inlier_ratio=tpl_cfg.get("min_inlier_ratio", 0.50),
                            min_area_px=tpl_cfg.get("min_area_px", 5000),
                        )
                        print(f"[court] compare: template tracker "
                              f"with {keypoint_tracker_a.active_count} keypoints")
                elif tracker_type == "template":
                    from src.court.template_tracker import TemplateKeypointTracker

                    keypoint_tracker = TemplateKeypointTracker(
                        template_half_size=tpl_cfg.get("template_half_size", 15),
                        search_radius=tpl_cfg.get("search_radius", 50),
                        min_ncc_score=tpl_cfg.get("min_ncc_score", 0.60),
                        recovery_radius=tpl_cfg.get("recovery_radius", 120),
                        max_lost_frames=tpl_cfg.get("max_lost_frames", 5),
                        smooth_alpha=tpl_cfg.get("smooth_alpha", 0.50),
                        max_jump_px=tpl_cfg.get("max_jump_px", 45),
                        geometry_consistency_enabled=tpl_cfg.get("geometry_consistency_enabled", True),
                        geometry_ransac_px=tpl_cfg.get("geometry_ransac_px", 6.0),
                        geometry_snap_px=tpl_cfg.get("geometry_snap_px", 12.0),
                    )
                    keypoint_tracker.init(ref_frame, dict(court_observation.keypoints))
                    kp_tracking_mode = "template"
                    print(f"[court] template tracker v2 initialised on frame {calib_ref_frame} "
                          f"with {keypoint_tracker.active_count} keypoints "
                          f"(alpha={keypoint_tracker._smooth_alpha}, "
                          f"max_jump={keypoint_tracker._max_jump}px, "
                          f"search_radius={keypoint_tracker._search_radius})")

                    # Compare mode: build second tracker with different params
                    keypoint_tracker_a = None
                    court_projector_a = None
                    if args.compare:
                        keypoint_tracker_a = TemplateKeypointTracker(
                            template_half_size=tpl_cfg.get("template_half_size", 15),
                            search_radius=tpl_cfg.get("compare_search_radius", 50),
                            min_ncc_score=tpl_cfg.get("compare_min_ncc_score",
                                                      tpl_cfg.get("min_ncc_score", 0.60)),
                            recovery_radius=tpl_cfg.get("recovery_radius", 120),
                            max_lost_frames=tpl_cfg.get("max_lost_frames", 5),
                            smooth_alpha=tpl_cfg.get("compare_smooth_alpha",
                                                     tpl_cfg.get("smooth_alpha", 0.50)),
                            max_jump_px=tpl_cfg.get("compare_max_jump_px",
                                                    tpl_cfg.get("max_jump_px", 45)),
                            geometry_consistency_enabled=tpl_cfg.get("geometry_consistency_enabled", True),
                            geometry_ransac_px=tpl_cfg.get("geometry_ransac_px", 6.0),
                            geometry_snap_px=tpl_cfg.get("geometry_snap_px", 12.0),
                        )
                        keypoint_tracker_a.init(ref_frame, dict(court_observation.keypoints))
                        court_projector_a = CourtProjector(
                            max_reproj_error_px=tpl_cfg.get("max_reproj_error_px", 15.0),
                            min_inlier_ratio=tpl_cfg.get("min_inlier_ratio", 0.50),
                            min_area_px=tpl_cfg.get("min_area_px", 5000),
                        )
                        print(f"[court] compare tracker initialised "
                              f"with {keypoint_tracker_a.active_count} keypoints")
                elif tracker_type == "lk":
                    from src.court.lk_tracker import LKKeypointTracker

                    lk_win = tpl_cfg.get("lk_win_size", 31)
                    keypoint_tracker = LKKeypointTracker(
                        win_size=(lk_win, lk_win),
                        max_level=tpl_cfg.get("lk_max_level", 3),
                    )
                    keypoint_tracker.init(ref_frame, dict(court_observation.keypoints))
                    kp_tracking_mode = "lk"
                    print(f"[court] LK optical-flow tracker initialised on frame {calib_ref_frame} "
                          f"with {keypoint_tracker.active_count} keypoints "
                          f"(win={lk_win}x{lk_win})")

                else:
                    print(f"[court] unknown tracker type '{tracker_type}' — "
                          f"court tracking disabled")
                    kp_tracking_mode = "none"

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[court] tracking will activate from frame {calib_ref_frame + 1} onwards")
            else:
                print(f"[court] WARNING: cannot read frame {calib_ref_frame} for tracker init")
                kp_tracking_mode = "static"
        elif court_observation is not None and kp_tracking_mode != "perframe":
            print(f"[court] only {court_observation.num_visible} keypoints — "
                  f"need ≥4 for projection, using static mode")
            kp_tracking_mode = "static"

        if court_observation is not None and tracker_type in ("static", "none", "fixed"):
            kp_tracking_mode = "static"
            print(f"[court] fixed homography mode: reuse calibration from frame {calib_ref_frame}")

        court_cfg = cfg.get("court", {})
        court_projector = CourtProjector(
            allow_cached=court_cfg.get("allow_cached_homography", True),
            max_cached_frames=court_cfg.get("max_cached_frames", 30),
            min_inlier_ratio=court_cfg.get("min_inlier_ratio", 0.50),
            max_reproj_error_px=court_cfg.get("max_reproj_error_px", 8.0),
            min_area_px=court_cfg.get("min_area_px", 5000),
        )

    pipeline = SideViewPipeline(
        detector=detector,
        tracker=tracker,
        event_detector=event_detector,
        court_layout=court_layout,
        court_projector=court_projector,
        court_observation=court_observation if kp_tracking_mode == "static" else None,
        keypoint_tracker=keypoint_tracker,
        player_detector=player_detector,
        detector_interval=cfg.get("tracking", {}).get("detector_interval", 1),
    )

    renderer = OverlayRenderer(
        trail_length=args.trail,
        court_renderer=court_renderer,
        court_layout=court_layout,
        court_projector=court_projector,
        draw_image_court=cfg.get("court", {}).get("draw_image_court_when_projected", True),
        draw_mini_court=cfg.get("court", {}).get("draw_mini_court", True),
        static_court_observation=(
            court_observation if kp_tracking_mode == "static" else None
        ),
    )

    detector_label = _detector_label(detector)

    print(f"[pipeline] detector={detector_label}  "
          f"tracker=simple  "
          f"court={'enabled' if court_layout else 'disabled'}  "
          f"kp_mode={kp_tracking_mode}  "
          f"players={'enabled' if cfg.get('players', {}).get('enabled') else 'disabled'}")
    if args.preview:
        preview_scale = max(0.1, min(1.0, args.display_scale))
        print(f"[preview] 实时预览已开启 | 按 Q 退出 | 按 空格 暂停/继续")
        print(f"[preview] 显示缩放: {preview_scale:.0%} ({int(width*preview_scale)}x{int(height*preview_scale)})")
        cv2.namedWindow("Side-View Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Side-View Analysis", int(width * preview_scale), int(height * preview_scale))
    print(f"[pipeline] processing …")

    # --- main loop -------------------------------------------------------
    start_frame = args.start_frame
    if start_frame is None and kp_tracking_mode in ("optical_flow", "template", "pose", "lk", "hybrid", "perframe", "static"):
        start_frame = calib_ref_frame  # default to where tracking begins
    if start_frame is None:
        start_frame = 0
    start_frame = max(0, start_frame)

    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[pipeline] starting from frame {start_frame}")

    frame_idx = start_frame
    max_frames = (
        min(start_frame + args.max_frames, total_frames)
        if args.max_frames is not None
        else total_frames
    )
    t_start = time.perf_counter()
    court_runtime_cfg = cfg.get("court", {})
    enable_auto_recalib = court_runtime_cfg.get("auto_recalibrate_on_drift", False)
    enable_template_refresh = court_runtime_cfg.get("refresh_templates_on_stable", False)
    _recalib_cooldown = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps_in if fps_in > 0 else None

        # Per-frame calibration mode: re-detect every frame, no tracking
        if kp_tracking_mode == "perframe":
            from tools.detect_court_cv import detect_court_single_frame
            kps_new = detect_court_single_frame(frame, downscale=2)
            if kps_new is not None and len(kps_new) >= 4:
                pipeline.court_observation = CourtObservation.from_calibration(kps_new)
            elif frame_idx == start_frame:
                print(f"  [perframe] frame {frame_idx}: detection failed")

        # Keep runtime interventions explicit and opt-in.
        if keypoint_tracker is not None and hasattr(keypoint_tracker, 'quality_report'):
            q = keypoint_tracker.quality_report()
            hold_count = q.get("hold_count", 0)
            pose_mode = q.get("pose_mode", "?")
            line_score = q.get("last_line_score", 0.0)

            if enable_auto_recalib and hold_count > 15 and _recalib_cooldown <= 0:
                from tools.detect_court_cv import detect_court_single_frame
                kps_new = detect_court_single_frame(frame, downscale=2)
                if kps_new is not None and len(kps_new) >= 4:
                    keypoint_tracker.reset()
                    keypoint_tracker.init(frame, dict(kps_new))
                    calib_ref_frame = frame_idx
                    _recalib_cooldown = 20
                    print(f"  [re-calib] frame {frame_idx}: "
                          f"hold={hold_count} -> {keypoint_tracker.active_count} kp")

            elif enable_template_refresh and hasattr(keypoint_tracker, 'refresh_templates'):
                if pose_mode == "observed" and line_score > 0.5:
                    keypoint_tracker.refresh_templates(frame)

            if (args.detect_every > 0 and frame_idx > start_frame
                    and (frame_idx - calib_ref_frame) % args.detect_every == 0
                    and _recalib_cooldown <= 0):
                from tools.detect_court_cv import detect_court_single_frame
                kps_new = detect_court_single_frame(frame, downscale=2)
                if kps_new is not None and len(kps_new) >= 4:
                    keypoint_tracker.reset()
                    keypoint_tracker.init(frame, dict(kps_new))
                    calib_ref_frame = frame_idx
                    print(f"  [re-detect] frame {frame_idx}: "
                          f"{keypoint_tracker.active_count} kp tracked")

            if _recalib_cooldown > 0:
                _recalib_cooldown -= 1

        result = pipeline.process_frame(frame, frame_idx, timestamp_s)

        # Compare mode: run baseline tracker in parallel
        obs_a = None
        if keypoint_tracker_a is not None:
            obs_a = keypoint_tracker_a.update(frame)
            court_projector_a.update(obs_a)

        # JSONL
        if jsonl_fh is not None:
            jsonl_fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        # Overlay (main version B)
        frame = renderer.draw(frame, result, detector_label)

        # Compare overlay: version A (baseline) in yellow/cyan
        if obs_a is not None and court_projector_a is not None:
            _draw_compare_overlay(frame, obs_a, court_projector_a, court_layout,
                                  result, keypoint_tracker, keypoint_tracker_a)

        # Output video
        if vw is not None:
            vw.write(frame)

        # Preview (default: on, press Q to quit, Space to pause)
        if args.preview:
            # Draw keyboard hints on the preview frame
            hint_y = height - 30
            cv2.putText(frame, "Q:退出  空格:暂停/继续", (width - 230, hint_y),
                       FONT, 0.4, C_GREY, 1, cv2.LINE_AA)

            # Scale frame for display (original kept intact for video output)
            if preview_scale != 1.0:
                dw = int(width * preview_scale)
                dh = int(height * preview_scale)
                display_frame = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            cv2.imshow("Side-View Analysis", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[preview] 用户按 Q 退出")
                break
            elif key == ord(" "):
                print("[preview] 已暂停，按任意键继续…")
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 == ord("q"):
                        print("[preview] 用户按 Q 退出")
                        frame_idx = max_frames  # force exit
                        break
                    elif k2 == ord(" "):
                        print("[preview] 继续播放")
                        break
                if frame_idx >= max_frames:
                    break

        frame_idx += 1

    # --- cleanup ---------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    cap.release()
    if vw is not None:
        vw.release()
    if jsonl_fh is not None:
        jsonl_fh.close()
    cv2.destroyAllWindows()

    # --- summary ---------------------------------------------------------
    detector_failed = getattr(detector, "disabled", False)
    processed = frame_idx - start_frame

    print(f"\n[done] {processed} frames in {elapsed:.1f}s  "
          f"({processed / elapsed:.1f} fps avg)  "
          f"[{start_frame}..{frame_idx - 1}]")
    if detector_failed:
        print(f"       WARNING: detector failed — output is null-detector only")
    if out_video:
        print(f"       overlay → {out_video}")
    if out_jsonl:
        print(f"       jsonl   → {out_jsonl}")


def _draw_compare_overlay(frame, obs_a, proj_a, court_layout, result_b, tracker_b, tracker_a):
    """Draw baseline version (A) in yellow/cyan alongside main version (B)."""
    h, w = frame.shape[:2]
    q_a = tracker_a.quality_report()
    q_b = tracker_b.quality_report()

    # Version A keypoints — orange
    for kp_idx, (px, py) in obs_a.keypoints.items():
        cv2.circle(frame, (int(px), int(py)), 4, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, str(kp_idx), (int(px) + 5, int(py) - 3),
                    FONT, 0.3, (0, 165, 255), 1, cv2.LINE_AA)

    # Version A court lines — orange
    if proj_a.is_available and court_layout is not None:
        H = proj_a.homography
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = None
        if H_inv is not None:
            for i, j in court_layout.line_segments:
                ft1 = court_layout.canonical_keypoints[i]
                ft2 = court_layout.canonical_keypoints[j]
                try:
                    p1 = cv2.perspectiveTransform(
                        np.array([[[ft1[0], ft1[1]]]], dtype=np.float32), H_inv)
                    p2 = cv2.perspectiveTransform(
                        np.array([[[ft2[0], ft2[1]]]], dtype=np.float32), H_inv)
                    cv2.line(frame,
                             (int(p1[0][0][0]), int(p1[0][0][1])),
                             (int(p2[0][0][0]), int(p2[0][0][1])),
                             (0, 165, 255), 1, cv2.LINE_AA)
                except cv2.error:
                    pass

    # Legend
    err_a = proj_a.quality.get("mean_reproj_error_px", "?")
    err_b = result_b.diagnostics.get("projection_quality", {}).get("mean_reproj_error_px", "?")
    active_a = q_a.get("active_points", "?")
    active_b = q_b.get("active_points", "?")
    lost_a = q_a.get("total_lost", q_a.get("h_holds", "?"))
    lost_b = q_b.get("total_lost", q_b.get("h_holds", "?"))
    cv2.rectangle(frame, (w - 280, h - 50), (w - 10, h - 10), (40, 40, 40), -1)
    cv2.putText(frame, f"A(NCC) active={active_a} lost={lost_a} err={err_a}",
                (w - 270, h - 32), FONT, 0.35, (0, 165, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"B(NCC) active={active_b} lost={lost_b} err={err_b}",
                (w - 270, h - 16), FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA)


def _detector_label(detector: BallDetector) -> str:
    name = type(detector).__name__
    if name == "NullBallDetector":
        return "null"
    if name == "YoloBallDetector":
        if getattr(detector, "disabled", False):
            return "yolo(failed)"
        return "yolo"
    return name.lower()


if __name__ == "__main__":
    main()
