"""
Annotated court + ball video.

Usage:
  python tools/annotate_video.py                          # allframe + EMA smooth
  python tools/annotate_video.py --mode hybrid --recalib 5
"""

import argparse
import sys
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.detect_court_cv import (
    segment_court_v2, _raw_white_mask, detect_court_hybrid,
)
from src.court.template_tracker import TemplateKeypointTracker

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/samples/test.mp4")
parser.add_argument("--output", default="outputs/court_annotated.mp4")
parser.add_argument("--start", type=int, default=300)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--recalib", type=int, default=5)
parser.add_argument("--mode", default="allframe", choices=["hybrid", "allframe"])
parser.add_argument("--smooth", type=float, default=0.35,
                    help="EMA alpha (lower=more smooth)")
parser.add_argument("--ball-model", default="artifacts/models/ball_best.pt")
parser.add_argument("--ball-conf", type=float, default=0.25)
parser.add_argument("--no-ball", action="store_true")
parser.add_argument("--exclude", type=int, nargs=4, default=[0, 0, 120, 100],
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="Ball detection exclusion zone")
args = parser.parse_args()
EXCLUDE = args.exclude  # [x1, y1, x2, y2]

cap = cv2.VideoCapture(args.input)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
end_frame = args.end if args.end is not None else total_frames

# ── Ball detector ──
ball_model = None
if not args.no_ball:
    try:
        from ultralytics import YOLO
        ball_model = YOLO(args.ball_model)
        print(f"Ball model: {args.ball_model}")
    except Exception as e:
        print(f"Ball model unavailable: {e}")

print(f"Video: {args.input}  {w}x{h}  {fps:.1f}fps")
print(f"Frames: {args.start}..{end_frame}  mode={args.mode}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(args.output, fourcc, fps, (w, h + 26))
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

# ── State ──
tracker = None
n_ok, n_full, n_track, n_hold = 0, 0, 0, 0
last_recalib = -999
fi = args.start
cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
ema_kps = {}   # type: dict[int, tuple[float, float]]
ball_trail = deque(maxlen=20)

def _detect_frame(frame):
    """Full detection (downscale=2 for speed). Returns dict[int→(x,y)] or None."""
    h_s, w_s = h // 2, w // 2
    frame_ds = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_AREA)
    court_mask_ds, _ = segment_court_v2(frame_ds)
    if court_mask_ds is None:
        return None
    court_mask = cv2.resize(court_mask_ds, (w, h), interpolation=cv2.INTER_NEAREST)
    mr = np.where(court_mask.sum(axis=1) > 0)[0]
    y0 = max(0, mr[0] - 5) if len(mr) > 0 else 0
    y1 = min(h, mr[-1] + 5) if len(mr) > 0 else h
    geom_roi = np.zeros((h, w), dtype=np.uint8)
    geom_roi[y0:y1, :] = 255
    white_clipped = cv2.bitwise_and(_raw_white_mask(frame), geom_roi)
    result = detect_court_hybrid(frame, court_mask, white_clipped, downscale=2)
    if result is None:
        return None
    return result[0]


def _detect_ball(frame):
    """Detect ball positions. Returns list of (x, y, conf)."""
    if ball_model is None:
        return []
    # Mask exclusion zone (logo in top-left corner)
    masked = frame.copy()
    x1e, y1e, x2e, y2e = EXCLUDE
    cv2.rectangle(masked, (x1e, y1e), (x2e, y2e), (0, 0, 0), -1)
    results = ball_model(masked, conf=args.ball_conf, verbose=False)
    balls = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # ball class
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    conf = float(box.conf[0])
                    # Skip detections inside exclusion zone
                    if x1e <= cx <= x2e and y1e <= cy <= y2e:
                        continue
                    balls.append((cx, cy, conf))
    return balls


def _smooth_kps(new_kps):  # new_kps: dict[int, tuple[float, float]]
    """EMA smoothing."""
    global ema_kps
    smoothed = {}
    alpha = args.smooth
    for idx, (nx, ny) in new_kps.items():
        if idx in ema_kps:
            ox, oy = ema_kps[idx]
            sx = alpha * nx + (1 - alpha) * ox
            sy = alpha * ny + (1 - alpha) * oy
        else:
            sx, sy = nx, ny
        ema_kps[idx] = (sx, sy)
        smoothed[idx] = (sx, sy)
    return smoothed


# ── Thread pool for parallel court + ball ──
_executor = ThreadPoolExecutor(max_workers=2)

# ── Main loop ──
while fi < end_frame:
    ok, frame = cap.read()
    if not ok:
        break

    # ── Court + ball detection ──
    if args.mode == "allframe":
        # Parallel: court and ball run simultaneously
        future_court = _executor.submit(_detect_frame, frame)
        future_ball = _executor.submit(_detect_ball, frame)
        kps = future_court.result()
        balls = future_ball.result()

        if kps is not None:
            kps_draw = _smooth_kps(kps)
            n_kps = len(kps_draw)
            mode = "ALL"
            n_full += 1
            n_ok += 1
        else:
            n_kps = 0
            mode = "LOST"
            n_hold += 1
    else:
        balls = _detect_ball(frame)
        if tracker is None or (fi - last_recalib) >= args.recalib:
            kps = _detect_frame(frame)
            if kps is not None:
                if tracker is None:
                    tracker = TemplateKeypointTracker(
                        template_half_size=15, search_radius=50,
                        min_ncc_score=0.60, recovery_radius=120,
                        max_lost_frames=5, smooth_alpha=0.50,
                        max_jump_px=45.0,
                        geometry_consistency_enabled=True,
                        geometry_ransac_px=6.0,
                        geometry_snap_px=12.0)
                tracker.reset()
                tracker.init(frame, dict(kps))
                last_recalib = fi
                n_full += 1
                kps_draw = kps
                n_kps = len(kps_draw)
                mode = "FULL"
                n_ok += 1
            else:
                n_kps = 0
                mode = "LOST"
        else:
            obs = tracker.update(frame)
            if obs is not None and obs.num_visible >= 4:
                n_track += 1
                kps_draw = obs.keypoints
                n_kps = len(kps_draw)
                mode = "TRACK"
                n_ok += 1
            else:
                n_hold += 1
                n_kps = 0
                mode = "LOST"

    # Ball trail update
    if balls:
        best = max(balls, key=lambda b: b[2])
        ball_trail.append((best[0], best[1]))

    # ── Draw ──
    vis = frame.copy()

    # Ball trail
    if len(ball_trail) >= 2:
        pts = list(ball_trail)
        for i in range(1, len(pts)):
            alpha = (i + 1) / len(pts)
            col = (0, int(140 + 115 * alpha), int(200 + 55 * alpha))
            cv2.line(vis, (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]), int(pts[i][1])), col, max(1, int(3 * alpha)), cv2.LINE_AA)

    # Ball marker
    if balls:
        bx, by, bc = max(balls, key=lambda b: b[2])
        cv2.circle(vis, (int(bx), int(by)), 8, (0, 255, 255), -1)
        cv2.circle(vis, (int(bx), int(by)), 9, (255, 255, 255), 2)
        cv2.putText(vis, f"{bc:.0%}", (int(bx) + 12, int(by) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Court keypoints + lines
    if n_kps > 0:
        for idx, (px, py) in kps_draw.items():
            cv2.circle(vis, (int(px), int(py)), 5, (0, 220, 255), -1)
            cv2.circle(vis, (int(px), int(py)), 6, (255, 255, 255), 1)
            cv2.putText(vis, str(idx), (int(px) + 7, int(py) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        line_pairs = [
            (0, 2, (0, 0, 255)), (11, 13, (0, 255, 0)),
            (3, 5, (0, 180, 255)), (8, 10, (0, 255, 200)),
            (6, 7, (255, 180, 0)),
            (0, 11, (180, 180, 180)), (2, 13, (180, 180, 180)),
            (3, 8, (180, 180, 180)), (5, 10, (180, 180, 180)),
        ]
        for ki, kj, col in line_pairs:
            if ki in kps_draw and kj in kps_draw:
                cv2.line(vis, (int(kps_draw[ki][0]), int(kps_draw[ki][1])),
                         (int(kps_draw[kj][0]), int(kps_draw[kj][1])), col, 1, cv2.LINE_AA)

    bar = np.zeros((26, w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    ball_str = f"ball: ({int(bx):d},{int(by):d})" if balls else "ball: -"
    mode_str = f"[{mode}+EMA]" if args.mode == "allframe" else f"[{mode}]"
    cv2.putText(bar, f"Frame {fi:04d}  |  {n_kps} kps  {mode_str}  |  {ball_str}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    if args.mode == "allframe":
        cv2.putText(bar, f"ok={n_full}  lost={n_hold}",
                    (w - 200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(bar, f"full={n_full} track={n_track} hold={n_hold}",
                    (w - 280, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    vw.write(np.vstack([bar, vis]))

    if fi % 60 == 0:
        print(f"  frame {fi:4d}: {n_kps} kps [{mode}]  balls={len(balls)}")

    fi += 1

cap.release()
vw.release()

total = n_full + n_track + n_hold
print(f"\nDone: {total} frames  ->  {args.output}")
if args.mode == "allframe":
    print(f"  detections: {n_full}  lost: {n_hold}")
else:
    print(f"  calibrations: {n_full}  tracked: {n_track}  lost: {n_hold}")
