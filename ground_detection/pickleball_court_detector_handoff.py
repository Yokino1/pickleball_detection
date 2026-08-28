"""Self-contained pickleball half-court detector for handoff.

Pipeline:
    white-line enhancement -> LSD chains -> line intersections ->
    selected geometric points -> five-white-line topology ->
    semantic half-court keypoints/lines/regions

Runtime dependencies:
    Python 3.10+, NumPy, OpenCV

CLI example:
    python tools/pickleball_court_detector_handoff.py \
        --inputs data/left_test.png \
        --output-dir outputs/pickleball_court_handoff \
        --downscale 2

Python API:
    result = detect_court_image(input_path, output_dir, downscale=2)

The script writes eight compact audit images, a reconstruction JSON, and a
summary JSON. Coordinates in the reconstruction JSON are restored to the
original image coordinate system.

Known input contract:
    The input filename must currently contain left/right or Chinese left/right
    so that half-court keypoints receive the correct semantic names.

This file has no imports from other project modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TARGET_LINE_SHORT_NAMES = {
    "top_sideline": "TOP",
    "bottom_sideline": "BOT",
    "left_baseline": "LB",
    "left_non_volley_line": "LNVZ",
    "net_line": "NET",
    "right_non_volley_line": "RNVZ",
    "right_baseline": "RB",
    "left_centerline": "LC",
    "right_centerline": "RC",
}

CANONICAL_KEYPOINTS_FT = {
    "kp0": [0.0, 0.0],
    "kp1": [10.0, 0.0],
    "kp2": [20.0, 0.0],
    "kp3": [0.0, 15.0],
    "kp4": [10.0, 15.0],
    "kp5": [20.0, 15.0],
    "kp6": [0.0, 22.0],
    "kp7": [20.0, 22.0],
    "kp8": [0.0, 29.0],
    "kp9": [10.0, 29.0],
    "kp10": [20.0, 29.0],
    "kp11": [0.0, 44.0],
    "kp12": [10.0, 44.0],
    "kp13": [20.0, 44.0],
}

FULL_COURT_LINE_SEGMENTS = [
    ("kp0", "kp2"),
    ("kp11", "kp13"),
    ("kp0", "kp11"),
    ("kp2", "kp13"),
    ("kp3", "kp5"),
    ("kp6", "kp7"),
    ("kp8", "kp10"),
    ("kp1", "kp4"),
    ("kp9", "kp12"),
]


# Image I/O, white-line enhancement, and centerline extraction.

def sample_ratio(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float, samples: int) -> float:
    h, w = mask.shape[:2]
    hits = 0
    total = 0
    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        if 0 <= x < w and 0 <= y < h:
            total += 1
            if mask[y, x] > 0:
                hits += 1
    return hits / max(total, 1)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return frame


def read_calibration_frame(path: Path) -> tuple[np.ndarray, dict]:
    """Read one still image or frame 0 from a fixed-camera calibration video."""
    video_suffixes = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
    if path.suffix.lower() not in video_suffixes:
        return read_image(path), {
            "input_kind": "image",
            "input_frame_index": 0,
        }

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open calibration video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame 0 from calibration video: {path}")
    return frame, {
        "input_kind": "video",
        "input_frame_index": 0,
        "source_video_fps": fps,
        "source_video_frame_count": frame_count,
    }


def save_vis(path: Path, image: np.ndarray, max_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = image
    if max_width > 0 and out.shape[1] > max_width:
        scale = max_width / float(out.shape[1])
        out = cv2.resize(
            out,
            (max_width, int(round(out.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), out)


def mask_debug_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    fill_color: tuple[int, int, int] = (255, 0, 255),
    edge_color: tuple[int, int, int] = (0, 255, 255),
    title: str = "binary mask debug overlay",
) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    dark = cv2.convertScaleAbs(frame, alpha=0.42, beta=0)
    color_layer = np.zeros_like(frame)
    color_layer[mask_u8 > 0] = fill_color
    vis = dark.copy()
    vis[mask_u8 > 0] = cv2.addWeighted(dark[mask_u8 > 0], 0.25, color_layer[mask_u8 > 0], 0.75, 0)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(vis, contours, -1, edge_color, 2, cv2.LINE_AA)
    count = int(np.count_nonzero(mask_u8))
    cv2.rectangle(vis, (8, 8), (min(vis.shape[1] - 8, 760), 52), (0, 0, 0), -1)
    cv2.putText(vis, f"{title}  pixels={count}", (18, 39),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def normalize_u8(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None and np.any(mask > 0):
        values = image[mask > 0]
    else:
        values = image.reshape(-1)
    lo, hi = np.percentile(values, [2, 99]) if values.size else (0, 255)
    if hi <= lo + 1e-6:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    out = (image.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def make_line_kernel(length: int, thickness: int, angle_deg: float) -> np.ndarray:
    size = int(length)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    radius = max(1, size // 2 - 1)
    angle = np.deg2rad(angle_deg)
    dx = int(round(np.cos(angle) * radius))
    dy = int(round(np.sin(angle) * radius))
    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness, cv2.LINE_AA)
    return kernel


def centerline_from_binary(binary: np.ndarray, roi: np.ndarray) -> np.ndarray:
    mask = ((binary > 0) & (roi > 0)).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return mask
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    local_max = dist >= cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) - 1e-3
    center = np.zeros_like(mask)
    center[(local_max) & (dist >= 1.0)] = 255
    return cv2.bitwise_and(center, roi)


def clean_centerline_for_line_fit(centerline: np.ndarray, roi: np.ndarray) -> np.ndarray:
    mask = ((centerline > 0) & (roi > 0)).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return mask
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, 8)
    spans = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep_labels = (stats[:, cv2.CC_STAT_AREA] >= 5) | (spans >= 9)
    keep_labels[0] = False
    cleaned = keep_labels[labels].astype(np.uint8) * 255
    return cv2.bitwise_and(cleaned, roi)


def enhance_white_lines_for_lsd(frame: np.ndarray, roi: np.ndarray) -> dict[str, np.ndarray | float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_local = clahe.apply(gray)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    white_like = val * (1.0 - 0.65 * sat / 255.0)
    white_like = normalize_u8(white_like, roi)
    white_paint = np.zeros_like(gray)
    white_paint[(val >= 145.0) & (sat <= 105.0) & (roi > 0)] = 255
    white_paint = cv2.morphologyEx(white_paint, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    white_paint = cv2.morphologyEx(white_paint, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    bright_tophat = cv2.morphologyEx(gray_local, cv2.MORPH_TOPHAT, ellipse)

    line_tophat = np.zeros_like(gray_local)
    for angle in (0, 25, 55, 90, 125, 155):
        kernel = make_line_kernel(31, 3, angle)
        opened = cv2.morphologyEx(gray_local, cv2.MORPH_OPEN, kernel)
        response = cv2.subtract(gray_local, opened)
        line_tophat = np.maximum(line_tophat, response)

    response = (
        normalize_u8(line_tophat, roi).astype(np.float32) * 0.55
        + normalize_u8(bright_tophat, roi).astype(np.float32) * 0.25
        + white_like.astype(np.float32) * 0.20
    )
    response = normalize_u8(response, roi)
    response = cv2.bitwise_and(response, roi)

    values = response[roi > 0]
    if values.size:
        _otsu, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = max(float(_otsu), float(np.percentile(values, 86)))
    else:
        threshold = 80.0
    binary = np.zeros_like(response)
    binary[response >= threshold] = 255
    binary = cv2.bitwise_and(binary, roi)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    centerline = centerline_from_binary(binary, roi)
    centerline_clean = clean_centerline_for_line_fit(centerline, roi)
    lsd_input = centerline_clean
    return {
        "gray_local": gray_local,
        "white_like": white_like,
        "white_paint": white_paint,
        "line_tophat": normalize_u8(line_tophat, roi),
        "bright_tophat": normalize_u8(bright_tophat, roi),
        "response": response,
        "binary": binary,
        "centerline": centerline,
        "centerline_clean": centerline_clean,
        "lsd_input": lsd_input,
        "threshold": float(threshold),
    }


def upscale_gray(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


# LSD chaining, line fitting, intersections, and geometric point selection.

def angle_gap(a: float, b: float) -> float:
    gap = abs(float(a) - float(b)) % 180.0
    return min(gap, 180.0 - gap)


def line_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, int(np.argmax(vals))]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    rho = float(np.dot(center, normal))
    return center.astype(np.float32), direction.astype(np.float32), rho


def parse_lsd_segments(lines: np.ndarray, scale: float, min_len_ds: float) -> list[dict]:
    segments: list[dict] = []
    for idx, (x1, y1, x2, y2) in enumerate(lines.reshape(-1, 4)):
        length_ds = float(np.hypot(x2 - x1, y2 - y1))
        if length_ds < min_len_ds:
            continue
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 0:
            angle += 180.0
        segments.append({
            "id": idx,
            "x1": float(x1) * scale,
            "y1": float(y1) * scale,
            "x2": float(x2) * scale,
            "y2": float(y2) * scale,
            "cx": float((x1 + x2) * 0.5) * scale,
            "cy": float((y1 + y2) * 0.5) * scale,
            "x1_ds": float(x1),
            "y1_ds": float(y1),
            "x2_ds": float(x2),
            "y2_ds": float(y2),
            "cx_ds": float((x1 + x2) * 0.5),
            "cy_ds": float((y1 + y2) * 0.5),
            "length": length_ds * scale,
            "length_ds": length_ds,
            "angle": angle,
        })
    return segments


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def segment_pair_support(mask: np.ndarray, a: dict, b: dict) -> float:
    x1, y1 = a["cx_ds"], a["cy_ds"]
    x2, y2 = b["cx_ds"], b["cy_ds"]
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 4.0:
        return 1.0
    return sample_ratio(mask, x1, y1, x2, y2, max(8, int(dist / 3.0)))


def build_connectivity(segments: list[dict], centerline: np.ndarray) -> tuple[list[tuple[int, int]], list[list[dict]]]:
    h, w = centerline.shape[:2]
    max_center_gap = max(32.0, min(h, w) * 0.030)
    max_perp_gap = max(4.0, min(h, w) * 0.006)
    max_angle_gap = 14.0
    uf = UnionFind(len(segments))
    pairs: list[tuple[int, int]] = []
    cell_size = max_center_gap
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, seg in enumerate(segments):
        cell = (int(float(seg["cx_ds"]) // cell_size), int(float(seg["cy_ds"]) // cell_size))
        buckets.setdefault(cell, []).append(idx)

    for i, a in enumerate(segments):
        ca = np.array([a["cx_ds"], a["cy_ds"]], dtype=np.float32)
        va_angle = np.deg2rad(a["angle"])
        va = np.array([np.cos(va_angle), np.sin(va_angle)], dtype=np.float32)
        na = np.array([-va[1], va[0]], dtype=np.float32)
        cell_x = int(float(a["cx_ds"]) // cell_size)
        cell_y = int(float(a["cy_ds"]) // cell_size)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for j in buckets.get((cell_x + dx, cell_y + dy), []):
                    if j <= i:
                        continue
                    b = segments[j]
                    cb = np.array([b["cx_ds"], b["cy_ds"]], dtype=np.float32)
                    center_vec = cb - ca
                    center_dist = float(np.linalg.norm(center_vec))
                    if center_dist > max_center_gap:
                        continue
                    if angle_gap(a["angle"], b["angle"]) > max_angle_gap:
                        continue
                    perp_gap = abs(float(np.dot(center_vec, na)))
                    if perp_gap > max_perp_gap and min(a["length_ds"], b["length_ds"]) > 12.0:
                        continue
                    support = segment_pair_support(centerline, a, b)
                    if center_dist > 10.0 and support < 0.05:
                        continue
                    uf.union(i, j)
                    pairs.append((i, j))

    grouped: dict[int, list[dict]] = {}
    for idx, seg in enumerate(segments):
        grouped.setdefault(uf.find(idx), []).append(seg)
    chains = [items for items in grouped.values() if len(items) >= 2]
    return pairs, chains


def refit_chain(chain: list[dict], centerline: np.ndarray, chain_id: int) -> dict | None:
    pts = []
    for seg in chain:
        pts.append([seg["cx_ds"], seg["cy_ds"]])
        pts.append([seg["cx_ds"], seg["cy_ds"]])
        pts.append([seg["x1_ds"], seg["y1_ds"]])
        pts.append([seg["x2_ds"], seg["y2_ds"]])
    points = np.asarray(pts, dtype=np.float32)
    if len(points) < 4:
        return None
    center, direction, _rho = line_from_points(points)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    projections = (points - center) @ direction
    residuals = np.abs((points - center) @ normal)
    t1, t2 = float(np.min(projections)), float(np.max(projections))
    if t2 <= t1:
        return None
    p1 = center + t1 * direction
    p2 = center + t2 * direction
    span_ds = float(np.linalg.norm(p2 - p1))

    intervals = []
    for seg in chain:
        p_a = np.array([seg["x1_ds"], seg["y1_ds"]], dtype=np.float32)
        p_b = np.array([seg["x2_ds"], seg["y2_ds"]], dtype=np.float32)
        ta = float(np.dot(p_a - center, direction))
        tb = float(np.dot(p_b - center, direction))
        intervals.append((min(ta, tb), max(ta, tb)))
    intervals.sort()
    merged = []
    merge_gap = max(5.0, min(centerline.shape[:2]) * 0.006)
    for start, end in intervals:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    max_gap_ds = float(max(gaps)) if gaps else 0.0
    support_len_ds = float(sum(seg["length_ds"] for seg in chain))
    density = support_len_ds / max(span_ds, 1.0)
    line_support = sample_ratio(centerline, p1[0], p1[1], p2[0], p2[1], max(16, int(span_ds / 3.0)))
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle < 0:
        angle += 180.0
    return {
        "chain_id": chain_id,
        "x1_ds": float(p1[0]),
        "y1_ds": float(p1[1]),
        "x2_ds": float(p2[0]),
        "y2_ds": float(p2[1]),
        "angle": angle,
        "span_ds": span_ds,
        "support_len_ds": support_len_ds,
        "density": float(density),
        "line_support": float(line_support),
        "mean_residual_ds": float(np.mean(residuals)),
        "max_residual_ds": float(np.max(residuals)),
        "max_gap_ds": max_gap_ds,
        "segment_count": len(chain),
        "segments": chain,
    }


def filter_refit_lines(lines: list[dict], frame_shape: tuple[int, int], scale: float) -> list[dict]:
    h, w = frame_shape
    min_span = max(70.0 / scale, min(h, w) * 0.035)
    kept = []
    for line in lines:
        if line["span_ds"] < min_span:
            continue
        if line["mean_residual_ds"] > max(4.5, min(h, w) * 0.004):
            continue
        if line["max_gap_ds"] > max(42.0, line["span_ds"] * 0.32):
            continue
        if line["density"] < 0.18 and line["line_support"] < 0.08:
            continue
        kept.append(line)
    return sorted(kept, key=lambda item: (-item["span_ds"], -item["density"]))


def filter_ground_line_candidates(lines: list[dict], frame_shape: tuple[int, int]) -> list[dict]:
    h, _w = frame_shape
    kept = []
    for line in lines:
        y1 = float(line["y1_ds"])
        y2 = float(line["y2_ds"])
        center_y = (y1 + y2) * 0.5
        lower_y = max(y1, y2)
        angle = float(line["angle"])
        near_horizontal = min(angle_gap(angle, 0.0), angle_gap(angle, 180.0)) < 15.0

        if lower_y < h * 0.48:
            continue
        if center_y < h * 0.38:
            continue
        if near_horizontal and center_y < h * 0.43:
            continue
        kept.append(line)
    return sorted(kept, key=lambda item: (-item["span_ds"], -item["density"]))


def filter_near_halfcourt_line_candidates(lines: list[dict], frame_shape: tuple[int, int]) -> list[dict]:
    h, _w = frame_shape
    kept = []
    for line in lines:
        y1 = float(line["y1_ds"])
        y2 = float(line["y2_ds"])
        center_y = (y1 + y2) * 0.5
        lower_y = max(y1, y2)
        if lower_y < h * 0.10:
            continue
        if center_y < h * 0.06:
            continue
        kept.append(line)
    return sorted(kept, key=lambda item: (-item["span_ds"], -item["density"]))


def parse_hough_lines(lines: np.ndarray | None, centerline: np.ndarray, scale: float, base_id: int = 10000) -> list[dict]:
    if lines is None:
        return []
    h, w = centerline.shape[:2]
    parsed: list[dict] = []
    min_span = max(35.0, min(h, w) * 0.055)
    for idx, (x1, y1, x2, y2) in enumerate(lines.reshape(-1, 4)):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        span = float(np.hypot(dx, dy))
        if span < min_span:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < 0:
            angle += 180.0
        if angle_gap(angle, 90.0) < 18.0:
            continue
        center_y = (float(y1) + float(y2)) * 0.5
        if center_y < h * 0.045:
            continue
        line_support = sample_ratio(centerline, float(x1), float(y1), float(x2), float(y2), max(16, int(span / 3.0)))
        if line_support < 0.08:
            continue
        parsed.append({
            "chain_id": int(base_id + idx),
            "x1_ds": float(x1),
            "y1_ds": float(y1),
            "x2_ds": float(x2),
            "y2_ds": float(y2),
            "angle": angle,
            "span_ds": span,
            "support_len_ds": span,
            "density": 1.0,
            "line_support": float(line_support),
            "mean_residual_ds": 0.0,
            "max_residual_ds": 0.0,
            "max_gap_ds": 0.0,
            "segment_count": 1,
            "source": "hough_centerline",
        })
    return sorted(parsed, key=lambda item: (-item["span_ds"], -item["line_support"]))


def build_centerline_hough_lines(centerline: np.ndarray, scale: float) -> list[dict]:
    h, w = centerline.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    work = cv2.dilate(centerline, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        work,
        1,
        np.pi / 180.0,
        threshold=max(32, int(min(h, w) * 0.070)),
        minLineLength=max(32, int(min(h, w) * 0.055)),
        maxLineGap=max(18, int(min(h, w) * 0.040)),
    )
    return parse_hough_lines(lines, centerline, scale)


def merge_collinear_line_candidates(lines: list[dict], centerline: np.ndarray, base_id: int = 20000) -> list[dict]:
    if not lines:
        return []
    h, w = centerline.shape[:2]
    groups: list[dict] = []
    max_perp_gap = max(8.0, min(h, w) * 0.010)
    max_angle_gap = 7.0

    for line in sorted(lines, key=lambda item: -float(item["span_ds"])):
        point, direction = line_point_direction(line)
        placed = False
        for group in groups:
            if angle_gap(float(line["angle"]), float(group["angle"])) > max_angle_gap:
                continue
            normal = np.array([-group["direction"][1], group["direction"][0]], dtype=np.float32)
            perp_gap = abs(float(np.dot(point - group["point"], normal)))
            if perp_gap > max_perp_gap:
                continue
            group["lines"].append(line)
            pts = []
            for item in group["lines"]:
                pts.append([item["x1_ds"], item["y1_ds"]])
                pts.append([item["x2_ds"], item["y2_ds"]])
            fit_center, fit_dir, _rho = line_from_points(np.asarray(pts, dtype=np.float32))
            if float(fit_dir[0]) < 0:
                fit_dir = -fit_dir
            group["point"] = fit_center.astype(np.float32)
            group["direction"] = fit_dir.astype(np.float32)
            angle = float(np.degrees(np.arctan2(float(fit_dir[1]), float(fit_dir[0]))))
            if angle < 0:
                angle += 180.0
            group["angle"] = angle
            placed = True
            break
        if not placed:
            groups.append({
                "angle": float(line["angle"]),
                "point": point.astype(np.float32),
                "direction": direction.astype(np.float32),
                "lines": [line],
            })

    merged: list[dict] = []
    for idx, group in enumerate(groups):
        pts = []
        source_ids = []
        for line in group["lines"]:
            pts.append([line["x1_ds"], line["y1_ds"]])
            pts.append([line["x2_ds"], line["y2_ds"]])
            source_ids.append(int(line["chain_id"]))
        points = np.asarray(pts, dtype=np.float32)
        if len(points) < 2:
            continue
        center, direction, _rho = line_from_points(points)
        if float(direction[0]) < 0:
            direction = -direction
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        projections = (points - center) @ direction
        residuals = np.abs((points - center) @ normal)
        t1, t2 = float(np.min(projections)), float(np.max(projections))
        if t2 <= t1:
            continue
        p1 = center + t1 * direction
        p2 = center + t2 * direction
        span = float(np.linalg.norm(p2 - p1))
        if span < max(45.0, min(h, w) * 0.060):
            continue
        line_support = sample_ratio(centerline, p1[0], p1[1], p2[0], p2[1], max(16, int(span / 3.0)))
        source_support = sum(float(line.get("line_support", 0.0)) * float(line["span_ds"]) for line in group["lines"])
        source_span = sum(float(line["span_ds"]) for line in group["lines"])
        density = source_span / max(span, 1.0)
        angle = float(np.degrees(np.arctan2(float(direction[1]), float(direction[0]))))
        if angle < 0:
            angle += 180.0
        if line_support < 0.045 and source_support / max(source_span, 1.0) < 0.12:
            continue
        merged.append({
            "chain_id": int(base_id + idx),
            "x1_ds": float(p1[0]),
            "y1_ds": float(p1[1]),
            "x2_ds": float(p2[0]),
            "y2_ds": float(p2[1]),
            "angle": angle,
            "span_ds": span,
            "support_len_ds": float(source_span),
            "density": float(density),
            "line_support": float(line_support),
            "mean_residual_ds": float(np.mean(residuals)),
            "max_residual_ds": float(np.max(residuals)),
            "max_gap_ds": 0.0,
            "segment_count": len(group["lines"]),
            "source": "merged_lsd_hough",
            "source_line_ids": sorted(set(source_ids)),
        })
    return sorted(merged, key=lambda item: (-item["span_ds"], -item["density"]))


def line_point_direction(line: dict) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.array([line["x1_ds"], line["y1_ds"]], dtype=np.float32)
    p2 = np.array([line["x2_ds"], line["y2_ds"]], dtype=np.float32)
    direction = p2 - p1
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        angle = np.deg2rad(float(line.get("angle", 0.0)))
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    else:
        direction = direction / norm
    if direction[0] < 0:
        direction = -direction
    point = (p1 + p2) * 0.5
    return point, direction.astype(np.float32)


def line_intersection(line_a: dict, line_b: dict) -> tuple[float, float] | None:
    pa, va = line_point_direction(line_a)
    pb, vb = line_point_direction(line_b)
    cross = float(va[0] * vb[1] - va[1] * vb[0])
    if abs(cross) < 1e-5:
        return None
    delta = pb - pa
    t = float((delta[0] * vb[1] - delta[1] * vb[0]) / cross)
    p = pa + t * va
    return float(p[0]), float(p[1])


def line_projection_interval(line: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    point, direction = line_point_direction(line)
    a = np.array([line["x1_ds"], line["y1_ds"]], dtype=np.float32)
    b = np.array([line["x2_ds"], line["y2_ds"]], dtype=np.float32)
    ta = float(np.dot(a - point, direction))
    tb = float(np.dot(b - point, direction))
    return point, direction, min(ta, tb), max(ta, tb)


def point_overrun_from_observed_span(line: dict, point_xy: tuple[float, float]) -> float:
    point, direction, t_min, t_max = line_projection_interval(line)
    query = np.array(point_xy, dtype=np.float32)
    t = float(np.dot(query - point, direction))
    if t < t_min:
        return float(t_min - t)
    if t > t_max:
        return float(t - t_max)
    return 0.0


def local_extended_segment(
    line: dict,
    frame_shape: tuple[int, int],
    extend_ratio: float = 1.35,
) -> tuple[tuple[float, float], tuple[float, float]]:
    h, w = frame_shape
    point, direction, t_min, t_max = line_projection_interval(line)
    span = max(1.0, t_max - t_min)
    margin = min(span * extend_ratio, min(h, w) * 0.28)
    p1 = point + (t_min - margin) * direction
    p2 = point + (t_max + margin) * direction
    p1[0] = float(np.clip(p1[0], -w * 0.10, w * 1.10))
    p1[1] = float(np.clip(p1[1], -h * 0.10, h * 1.10))
    p2[0] = float(np.clip(p2[0], -w * 0.10, w * 1.10))
    p2[1] = float(np.clip(p2[1], -h * 0.10, h * 1.10))
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def group_lines_by_direction(lines: list[dict], max_groups: int = 4) -> list[dict]:
    groups: list[dict] = []
    for line in sorted(lines, key=lambda item: -float(item["span_ds"])):
        placed = False
        for group in groups:
            if angle_gap(float(line["angle"]), float(group["angle"])) <= 13.0:
                group["lines"].append(line)
                weights = np.array([max(1.0, float(item["span_ds"])) for item in group["lines"]], dtype=np.float32)
                angles = np.deg2rad([float(item["angle"]) * 2.0 for item in group["lines"]])
                mean_angle = 0.5 * np.arctan2(float(np.sum(np.sin(angles) * weights)), float(np.sum(np.cos(angles) * weights)))
                deg = float(np.degrees(mean_angle))
                if deg < 0:
                    deg += 180.0
                group["angle"] = deg
                placed = True
                break
        if not placed:
            groups.append({"angle": float(line["angle"]), "lines": [line]})
    groups.sort(key=lambda group: -sum(float(line["span_ds"]) for line in group["lines"]))
    return groups[:max_groups]


def generate_white_line_intersections(
    lines: list[dict],
    frame_shape: tuple[int, int],
    min_y_ratio: float = 0.34,
) -> tuple[list[dict], list[dict]]:
    h, w = frame_shape
    raw: list[dict] = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i + 1:]:
            gap = angle_gap(float(line_a["angle"]), float(line_b["angle"]))
            if gap < 16.0 or gap > 145.0:
                continue
            pt = line_intersection(line_a, line_b)
            if pt is None:
                continue
            x, y = pt
            if not (-w * 0.08 <= x <= w * 1.08 and h * min_y_ratio <= y <= h * 1.10):
                continue

            overrun_a = point_overrun_from_observed_span(line_a, pt)
            overrun_b = point_overrun_from_observed_span(line_b, pt)
            span_a = float(line_a["span_ds"])
            span_b = float(line_b["span_ds"])
            if overrun_a > max(300.0, span_a * 3.4):
                continue
            if overrun_b > max(300.0, span_b * 3.4):
                continue
            total_overrun = overrun_a + overrun_b
            if total_overrun > max(560.0, (span_a + span_b) * 2.4):
                continue

            observed_bonus = 220.0 if overrun_a < max(45.0, span_a * 0.25) and overrun_b < max(45.0, span_b * 0.25) else 0.0
            gap_score = 160.0 - abs(gap - 62.0) * 1.8
            score = (
                y * 0.72
                + (span_a + span_b) * 0.62
                + gap_score
                + observed_bonus
                - total_overrun * 0.42
            )
            raw.append({
                "point": (float(x), float(y)),
                "score": float(score),
                "line_ids": [int(line_a["chain_id"]), int(line_b["chain_id"])],
                "angles": [float(line_a["angle"]), float(line_b["angle"])],
                "angle_gap": float(gap),
                "overrun": [float(overrun_a), float(overrun_b)],
                "support_span": [span_a, span_b],
                "observed": bool(observed_bonus > 0.0),
            })

    raw.sort(key=lambda item: -float(item["score"]))
    clusters: list[dict] = []
    cluster_radius = max(16.0, min(h, w) * 0.012)
    for item in raw:
        x, y = item["point"]
        best_cluster = None
        best_dist = cluster_radius
        for cluster in clusters:
            cx, cy = cluster["point"]
            dist = float(np.hypot(x - cx, y - cy))
            if dist < best_dist:
                best_cluster = cluster
                best_dist = dist
        weight = max(1.0, float(item["score"]))
        if best_cluster is None:
            clusters.append({
                "point": (float(x), float(y)),
                "score": float(item["score"]),
                "members": 1,
                "weight": weight,
                "line_ids": sorted(set(item["line_ids"])),
                "best_line_ids": [int(value) for value in item["line_ids"]],
                "best_angle_gap": float(item["angle_gap"]),
                "best_overrun": [float(value) for value in item["overrun"]],
                "observed_members": 1 if item["observed"] else 0,
            })
        else:
            old_weight = float(best_cluster["weight"])
            new_weight = old_weight + weight
            cx, cy = best_cluster["point"]
            best_cluster["point"] = (
                float((cx * old_weight + x * weight) / new_weight),
                float((cy * old_weight + y * weight) / new_weight),
            )
            best_cluster["weight"] = float(new_weight)
            best_cluster["members"] = int(best_cluster["members"]) + 1
            best_cluster["line_ids"] = sorted(set(best_cluster["line_ids"]) | set(item["line_ids"]))
            best_cluster["observed_members"] = int(best_cluster["observed_members"]) + (1 if item["observed"] else 0)
            if float(item["score"]) > float(best_cluster["score"]):
                best_cluster["score"] = float(item["score"])
                best_cluster["best_line_ids"] = [int(value) for value in item["line_ids"]]
                best_cluster["best_angle_gap"] = float(item["angle_gap"])
                best_cluster["best_overrun"] = [float(value) for value in item["overrun"]]

    for cluster in clusters:
        cluster["cluster_score"] = float(cluster["score"]) + int(cluster["members"]) * 35.0 + int(cluster["observed_members"]) * 80.0
        cluster.pop("weight", None)
    clusters.sort(key=lambda item: -float(item["cluster_score"]))
    return raw, clusters


def point_to_line_distance(point: np.ndarray, line_point: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    delta = point - line_point
    t = float(np.dot(delta, direction))
    projected = line_point + t * direction
    dist = float(np.linalg.norm(point - projected))
    return dist, t


def line_direction_upward(line: dict) -> np.ndarray:
    _point, direction = line_point_direction(line)
    if float(direction[1]) > 0.0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def select_near_v_anchor(clusters: list[dict], lines: list[dict], frame_shape: tuple[int, int]) -> dict | None:
    h, w = frame_shape
    line_by_id = {int(line["chain_id"]): line for line in lines}
    candidates: list[tuple[float, int, dict]] = []
    for idx, cluster in enumerate(clusters):
        x, y = cluster["point"]
        if not (w * 0.04 <= x <= w * 0.96 and h * 0.58 <= y <= h * 1.03):
            continue
        best_line_ids = [int(value) for value in cluster["best_line_ids"]]
        best_lines = [line_by_id[line_id] for line_id in best_line_ids if line_id in line_by_id]
        if len(best_lines) < 2:
            continue
        if any(angle_gap(float(line["angle"]), 90.0) < 11.0 for line in best_lines):
            continue
        if not (34.0 <= float(cluster["best_angle_gap"]) <= 118.0):
            continue
        if int(cluster["observed_members"]) <= 0 and int(cluster["members"]) < 4:
            continue

        overrun_sum = float(sum(float(value) for value in cluster["best_overrun"]))
        support_span = sum(float(line["span_ds"]) for line in best_lines)
        if overrun_sum > max(520.0, support_span * 2.15):
            continue
        score = (
            float(y) * 1.4
            + int(cluster["members"]) * 45.0
            + int(cluster["observed_members"]) * 120.0
            - overrun_sum * 0.35
        )
        candidates.append((score, idx, cluster))
    if not candidates:
        return None
    _score, idx, cluster = max(candidates, key=lambda item: item[0])
    item = dict(cluster)
    item["id"] = int(idx)
    return item


def select_near_halfcourt_v_anchor(clusters: list[dict], lines: list[dict], frame_shape: tuple[int, int]) -> dict | None:
    h, w = frame_shape
    line_by_id = {int(line["chain_id"]): line for line in lines}
    candidates: list[tuple[float, int, dict]] = []
    for idx, cluster in enumerate(clusters):
        x, y = cluster["point"]
        if not (w * 0.06 <= x <= w * 0.92 and h * 0.22 <= y <= h * 0.90):
            continue
        if int(cluster["observed_members"]) <= 0:
            continue
        if not (42.0 <= float(cluster["best_angle_gap"]) <= 105.0):
            continue
        best_line_ids = [int(value) for value in cluster["best_line_ids"]]
        best_lines = [line_by_id[line_id] for line_id in best_line_ids if line_id in line_by_id]
        if len(best_lines) < 2:
            continue
        if any(angle_gap(float(line["angle"]), 90.0) < 16.0 for line in best_lines):
            continue
        overrun_sum = float(sum(float(value) for value in cluster["best_overrun"]))
        support_span = sum(float(line["span_ds"]) for line in best_lines)
        if overrun_sum > max(85.0, support_span * 0.28):
            continue
        score = (
            int(cluster["observed_members"]) * 190.0
            + int(cluster["members"]) * 35.0
            + support_span * 0.45
            + float(y) * 0.38
            - overrun_sum * 2.2
            - abs(float(x) - w * 0.38) * 0.08
        )
        candidates.append((score, idx, cluster))
    if not candidates:
        return select_near_v_anchor(clusters, lines, frame_shape)
    _score, idx, cluster = max(candidates, key=lambda item: item[0])
    item = dict(cluster)
    item["id"] = int(idx)
    return item


def build_v_rays_from_anchor(anchor: dict, lines: list[dict], frame_shape: tuple[int, int]) -> list[dict]:
    h, w = frame_shape
    anchor_pt = np.array(anchor["point"], dtype=np.float32)
    line_by_id = {int(line["chain_id"]): line for line in lines}
    candidate_ids = [int(value) for value in anchor["line_ids"]]
    candidate_lines = [line_by_id[line_id] for line_id in candidate_ids if line_id in line_by_id]
    if len(candidate_lines) < 2:
        candidate_lines = [line_by_id[line_id] for line_id in anchor["best_line_ids"] if line_id in line_by_id]

    ray_candidates: list[dict] = []
    for line in candidate_lines:
        direction = line_direction_upward(line)
        if float(direction[1]) > -0.05:
            continue
        dist, _t = point_to_line_distance(anchor_pt, *line_point_direction(line))
        if dist > max(36.0, min(h, w) * 0.024):
            continue
        side = "left_arm" if float(direction[0]) < 0.0 else "right_arm"
        ray_candidates.append({
            "side": side,
            "source_line_id": int(line["chain_id"]),
            "source_angle": float(line["angle"]),
            "point": anchor_pt.copy(),
            "direction": direction,
            "source_span_ds": float(line["span_ds"]),
            "distance_to_anchor_ds": float(dist),
        })

    selected: list[dict] = []
    for side in ("left_arm", "right_arm"):
        side_items = [item for item in ray_candidates if item["side"] == side]
        if side_items:
            selected.append(max(side_items, key=lambda item: (item["source_span_ds"], -item["distance_to_anchor_ds"])))

    if len(selected) < 2:
        ray_candidates.sort(key=lambda item: (-item["source_span_ds"], item["distance_to_anchor_ds"]))
        selected = []
        for item in ray_candidates:
            if all(angle_gap(float(np.degrees(np.arctan2(item["direction"][1], item["direction"][0]))), float(np.degrees(np.arctan2(old["direction"][1], old["direction"][0])))) > 28.0 for old in selected):
                selected.append(item)
            if len(selected) == 2:
                break

    corrected: list[dict] = []
    for idx, ray in enumerate(selected[:2]):
        compatible_points = [anchor_pt]
        direction = ray["direction"]
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        for line in lines:
            if angle_gap(float(line["angle"]), float(ray["source_angle"])) > 12.0:
                continue
            for x_key, y_key in (("x1_ds", "y1_ds"), ("x2_ds", "y2_ds")):
                pt = np.array([float(line[x_key]), float(line[y_key])], dtype=np.float32)
                delta = pt - anchor_pt
                t = float(np.dot(delta, direction))
                perp = abs(float(np.dot(delta, normal)))
                if -45.0 <= t <= max(w, h) * 1.8 and perp <= max(34.0, min(h, w) * 0.020):
                    compatible_points.append(pt)
        fit_points = np.asarray(compatible_points, dtype=np.float32)
        if len(fit_points) >= 3:
            _center, fit_dir, _rho = line_from_points(fit_points)
            if float(np.dot(fit_dir, direction)) < 0:
                fit_dir = -fit_dir
            if float(fit_dir[1]) <= -0.03:
                direction = fit_dir.astype(np.float32)
        corrected.append({
            "id": int(idx),
            "side": ray["side"],
            "source_line_id": int(ray["source_line_id"]),
            "source_angle": float(ray["source_angle"]),
            "point": anchor_pt.astype(np.float32),
            "direction": direction.astype(np.float32),
        })
    corrected.sort(key=lambda item: float(item["direction"][0]))
    return corrected


def select_near_side_intersections(
    clusters: list[dict],
    lines: list[dict],
    frame_shape: tuple[int, int],
    rays: list[dict],
    foreground_min_y_ratio: float = 0.47,
    require_ray_membership: bool = False,
) -> dict:
    h, w = frame_shape
    line_by_id = {int(line["chain_id"]): line for line in lines}
    anchor_y = float(rays[0]["point"][1]) if rays else None
    candidates: list[dict] = []
    rejected: list[dict] = []

    def reject(idx: int, cluster: dict, reason: str) -> None:
        rejected.append({
            "id": int(idx),
            "point": tuple(float(value) for value in cluster["point"]),
            "reason": reason,
        })

    for idx, cluster in enumerate(clusters):
        x, y = cluster["point"]
        if not (0.0 <= x <= w and h * foreground_min_y_ratio <= y <= h * 1.03):
            reject(idx, cluster, "outside_foreground_band")
            continue
        if int(cluster["members"]) <= 1 and int(cluster["observed_members"]) == 0:
            reject(idx, cluster, "single_weak_cluster")
            continue

        best_lines = [line_by_id[int(line_id)] for line_id in cluster["best_line_ids"] if int(line_id) in line_by_id]
        has_unsupported_vertical = (
            int(cluster["observed_members"]) == 0
            and any(angle_gap(float(line["angle"]), 90.0) < 11.0 for line in best_lines)
        )
        if has_unsupported_vertical:
            reject(idx, cluster, "unsupported_vertical_or_net_intersection")
            continue

        point = np.array([float(x), float(y)], dtype=np.float32)
        ray_distance = None
        for ray in rays:
            dist, t = point_to_line_distance(point, ray["point"], ray["direction"])
            if t >= -20.0:
                ray_distance = dist if ray_distance is None else min(ray_distance, dist)
        near_ray = ray_distance is not None and ray_distance <= max(32.0, min(h, w) * 0.020)
        if require_ray_membership and not near_ray:
            reject(idx, cluster, "not_on_near_halfcourt_v_rays")
            continue
        if anchor_y is not None and not near_ray and y > anchor_y + h * 0.045:
            reject(idx, cluster, "non_ray_point_below_v_anchor")
            continue
        if not near_ray and (x < w * 0.04 or x > w * 0.94):
            reject(idx, cluster, "non_ray_extreme_edge_point")
            continue
        candidates.append({
            "id": int(idx),
            "point": tuple(float(value) for value in cluster["point"]),
            "cluster": cluster,
            "near_ray": bool(near_ray),
            "ray_distance_ds": None if ray_distance is None else float(ray_distance),
        })

    duplicate_dx = max(70.0, w * 0.025)
    duplicate_dy = max(45.0, h * 0.026)

    def priority(item: dict) -> float:
        _x, y = item["point"]
        ray_penalty = min(140.0, float(item["ray_distance_ds"] or 140.0)) * 0.04
        return float(y) + (70.0 if item["near_ray"] else 0.0) - ray_penalty

    selected: list[dict] = []
    for item in sorted(candidates, key=priority, reverse=True):
        x, y = item["point"]
        duplicate_of = None
        for kept in selected:
            kx, ky = kept["point"]
            if abs(float(x) - float(kx)) <= duplicate_dx and abs(float(y) - float(ky)) <= duplicate_dy:
                duplicate_of = kept
                break
        if duplicate_of is not None:
            rejected.append({
                "id": int(item["id"]),
                "point": item["point"],
                "reason": f"near_side_duplicate_of_{duplicate_of['id']}",
            })
            continue
        selected.append(item)

    selected.sort(key=lambda item: int(item["id"]))
    rejected.sort(key=lambda item: int(item["id"]))
    return {"selected": selected, "rejected": rejected}


def infer_net_midpoint_from_selected(selected: list[dict], lines: list[dict], frame_shape: tuple[int, int]) -> dict | None:
    h, w = frame_shape
    vertical_lines = [line for line in lines if angle_gap(float(line["angle"]), 90.0) < 12.0 and float(line["span_ds"]) > min(h, w) * 0.05]
    if not vertical_lines or len(selected) < 2:
        return None
    net_line = max(vertical_lines, key=lambda line: float(line["span_ds"]) * (1.0 + float(line["line_support"])))
    net_x = float((net_line["x1_ds"] + net_line["x2_ds"]) * 0.5)
    points = [
        (int(item["id"]), np.array(item["point"], dtype=np.float32))
        for item in selected
        if item["point"][1] < h * 0.82
    ]
    best: dict | None = None
    for i, (id_a, p_a) in enumerate(points):
        for id_b, p_b in points[i + 1:]:
            dx = float(p_b[0] - p_a[0])
            if abs(dx) < w * 0.18:
                continue
            angle = abs(float(np.degrees(np.arctan2(float(p_b[1] - p_a[1]), dx))))
            if not (3.0 <= angle <= 22.0):
                continue
            t = (net_x - float(p_a[0])) / dx
            if not (-0.35 <= t <= 2.05):
                continue
            point = p_a + t * (p_b - p_a)
            if not (h * 0.48 <= float(point[1]) <= h * 0.82):
                continue
            score = float(point[1]) + float(np.linalg.norm(p_b - p_a)) * 0.08 - abs(t - 0.5) * 25.0
            if best is None or score > float(best["score"]):
                best = {
                    "point": (float(point[0]), float(point[1])),
                    "source_ids": [int(id_a), int(id_b)],
                    "net_line_id": int(net_line["chain_id"]),
                    "score": float(score),
                }
    return best


def detect_net_line_fallback(frame: np.ndarray, half_side: str) -> dict | None:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 145)
    hough = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(45, int(h * 0.08)),
        minLineLength=max(90, int(h * 0.24)),
        maxLineGap=max(24, int(h * 0.06)),
    )
    if hough is None:
        return None

    candidates = []
    for x1, y1, x2, y2 in hough.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < h * 0.24 or abs(dy) < abs(dx) * 2.6:
            continue
        x_mid = float(x1 + x2) * 0.5
        y_max = float(max(y1, y2))
        if y_max < h * 0.56:
            continue
        if half_side == "left":
            if x_mid < w * 0.72:
                continue
            side_score = x_mid / max(w, 1)
        else:
            if not (w * 0.12 <= x_mid <= w * 0.48):
                continue
            side_score = 1.0 - abs(x_mid - w * 0.28) / max(w, 1)
        score = length + h * 0.35 * side_score
        candidates.append((score, (float(x1), float(y1)), (float(x2), float(y2))))
    if not candidates:
        return None
    _score, p1, p2 = max(candidates, key=lambda item: item[0])
    midpoint = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
    return {
        "point": midpoint,
        "source_ids": [],
        "net_line_id": -1,
        "line_ds": [list(p1), list(p2)],
        "source": "vertical_net_edge_fallback",
    }


def infer_half_side(input_path: Path) -> str:
    stem = input_path.stem.lower()
    if "right" in stem or "右" in input_path.stem:
        return "right"
    if "left" in stem or "左" in input_path.stem:
        return "left"
    return "unknown"


# Compact audit visualization.

def display_px(value: float, display_scale: float) -> int:
    return max(1, int(round(float(value) / max(float(display_scale), 1e-6))))


def display_font(value: float, display_scale: float) -> float:
    return float(value) / max(float(display_scale), 1e-6)


def draw_white_line_intersections(
    frame: np.ndarray,
    lines: list[dict],
    raw: list[dict],
    clusters: list[dict],
    scale: float,
    frame_shape_ds: tuple[int, int],
) -> np.ndarray:
    vis = cv2.convertScaleAbs(frame, alpha=0.74, beta=0)
    for line in lines:
        (x1, y1), (x2, y2) = local_extended_segment(line, frame_shape_ds)
        cv2.line(
            vis,
            (int(round(x1 * scale)), int(round(y1 * scale))),
            (int(round(x2 * scale)), int(round(y2 * scale))),
            (0, 175, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            vis,
            (int(round(line["x1_ds"] * scale)), int(round(line["y1_ds"] * scale))),
            (int(round(line["x2_ds"] * scale)), int(round(line["y2_ds"] * scale))),
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

    for item in raw[:180]:
        x, y = item["point"]
        color = (0, 255, 255) if item["observed"] else (255, 160, 0)
        cv2.circle(vis, (int(round(x * scale)), int(round(y * scale))), 3, color, -1, cv2.LINE_AA)

    for idx, item in enumerate(clusters[:60]):
        x, y = item["point"]
        px = int(round(x * scale))
        py = int(round(y * scale))
        radius = 17 if idx < 24 else 10
        color = (0, 0, 255) if idx < 12 else (0, 255, 255)
        cv2.circle(vis, (px, py), radius + 3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)
        if idx < 24:
            label_pos = (px + 22, py - 22)
            cv2.putText(vis, f"{idx}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(vis, f"{idx}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(vis, f"{idx}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.45, color, 3, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"08 white-line intersections  raw={len(raw)} clusters={len(clusters)}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(vis, "green=white-line hypotheses  red=top clusters  cyan/orange=raw pair intersections", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_selected_intersections(
    frame: np.ndarray,
    lines: list[dict],
    selection: dict,
    inferred_net: dict | None,
    scale: float,
) -> np.ndarray:
    vis = cv2.convertScaleAbs(frame, alpha=0.76, beta=0)
    for line in lines:
        cv2.line(
            vis,
            (int(round(line["x1_ds"] * scale)), int(round(line["y1_ds"] * scale))),
            (int(round(line["x2_ds"] * scale)), int(round(line["y2_ds"] * scale))),
            (45, 115, 45),
            2,
            cv2.LINE_AA,
        )

    for item in selection["rejected"]:
        x, y = item["point"]
        px = int(round(float(x) * scale))
        py = int(round(float(y) * scale))
        cv2.circle(vis, (px, py), 12, (0, 0, 120), 2, cv2.LINE_AA)
        cv2.line(vis, (px - 10, py - 10), (px + 10, py + 10), (0, 0, 180), 3, cv2.LINE_AA)
        cv2.line(vis, (px - 10, py + 10), (px + 10, py - 10), (0, 0, 180), 3, cv2.LINE_AA)
        cv2.putText(vis, f"{item['id']}", (px + 15, py - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, f"{item['id']}", (px + 15, py - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (210, 210, 210), 2, cv2.LINE_AA)

    for item in selection["selected"]:
        x, y = item["point"]
        px = int(round(float(x) * scale))
        py = int(round(float(y) * scale))
        color = (0, 255, 0) if item["near_ray"] else (0, 230, 255)
        cv2.circle(vis, (px, py), 18, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 14, color, -1, cv2.LINE_AA)
        cv2.putText(vis, f"{item['id']}", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(vis, f"{item['id']}", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, f"{item['id']}", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.35, color, 2, cv2.LINE_AA)

    if inferred_net is not None:
        x, y = inferred_net["point"]
        px = int(round(float(x) * scale))
        py = int(round(float(y) * scale))
        cv2.circle(vis, (px, py), 18, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 13, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, "NET*", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(vis, "NET*", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, "NET*", (px + 22, py - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"09 selected intersection clusters  keep={len(selection['selected'])} reject={len(selection['rejected'])}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(vis, "green/yellow=kept  red X=rejected  NET*=inferred from selected line and net line", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_reconstruction_network(
    frame: np.ndarray,
    reconstruction: dict,
    scale: float,
    display_scale: float = 1.0,
) -> np.ndarray:
    vis = cv2.convertScaleAbs(frame, alpha=0.78, beta=0)
    h, w = vis.shape[:2]
    line_width = display_px(4, display_scale)
    family_colors = {"A": (255, 210, 0), "B": (0, 150, 255)}
    lines = reconstruction.get("grid_lines") or reconstruction.get("structural_lines", [])

    for line in lines:
        (x1, y1), (x2, y2) = line["segment_ds"]
        p1 = np.asarray([float(x1) * scale, float(y1) * scale], dtype=np.float64)
        p2 = np.asarray([float(x2) * scale, float(y2) * scale], dtype=np.float64)
        direction = p2 - p1
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction /= norm
        center = (p1 + p2) * 0.5
        extent = float(max(h, w) * 2)
        a = tuple(int(round(value)) for value in center - direction * extent)
        b = tuple(int(round(value)) for value in center + direction * extent)
        clipped, ca, cb = cv2.clipLine((0, 0, w, h), a, b)
        if not clipped:
            continue
        color = family_colors.get(str(line.get("family", "")), (0, 255, 255))
        cv2.line(vis, ca, cb, (15, 15, 15), line_width + display_px(3, display_scale), cv2.LINE_AA)
        cv2.line(vis, ca, cb, color, line_width, cv2.LINE_AA)
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        cv2.putText(
            vis,
            str(line["id"]),
            (mid[0] + display_px(5, display_scale), mid[1] - display_px(5, display_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            display_font(0.48, display_scale),
            color,
            display_px(2, display_scale),
            cv2.LINE_AA,
        )

    for point in reconstruction.get("selected_points", []):
        x, y = point["point_ds"]
        center = (int(round(float(x) * scale)), int(round(float(y) * scale)))
        cv2.circle(vis, center, display_px(8, display_scale), (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, center, display_px(8, display_scale), (15, 15, 15), display_px(2, display_scale), cv2.LINE_AA)

    text = (
        f"court-relative line families  side={reconstruction['half_side']}  "
        f"A={reconstruction['counts']['family_a_lines']}  "
        f"B={reconstruction['counts']['family_b_lines']}  "
        f"grid={reconstruction['counts']['grid_lines']}"
    )
    cv2.putText(vis, text, (display_px(16, display_scale), display_px(30, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.50, display_scale), (0, 0, 0), display_px(4, display_scale), cv2.LINE_AA)
    cv2.putText(vis, text, (display_px(16, display_scale), display_px(30, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.50, display_scale), (255, 255, 255), display_px(2, display_scale), cv2.LINE_AA)
    return vis


def draw_completed_grid(
    frame: np.ndarray,
    reconstruction: dict,
    scale: float,
    display_scale: float = 1.0,
) -> np.ndarray:
    vis = draw_reconstruction_network(frame, reconstruction, scale, display_scale)
    source_colors = {
        "selected": (0, 255, 0),
        "raw_cluster": (0, 255, 255),
        "line_intersection": (255, 255, 0),
    }
    for point in reconstruction.get("grid_intersections", []):
        x, y = point["point_ds"]
        px = int(round(float(x) * scale))
        py = int(round(float(y) * scale))
        color = source_colors.get(str(point.get("source", "")), (255, 255, 255))
        radius = display_px(8, display_scale)
        cv2.circle(vis, (px, py), radius + display_px(3, display_scale), (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            str(point["id"]),
            (px + display_px(10, display_scale), py - display_px(8, display_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            display_font(0.50, display_scale),
            color,
            display_px(2, display_scale),
            cv2.LINE_AA,
        )
    return vis


def draw_halfcourt_result(
    frame: np.ndarray,
    reconstruction: dict,
    scale: float,
    display_scale: float = 1.0,
) -> np.ndarray:
    vis = cv2.convertScaleAbs(frame, alpha=0.82, beta=0)
    overlay = vis.copy()
    region_colors = [
        (30, 190, 255),
        (30, 220, 120),
        (255, 170, 40),
    ]
    region_index = 0
    for region in reconstruction.get("target_regions", {}).values():
        if region.get("status") != "detected":
            continue
        pts = np.asarray(
            [
                [int(round(float(x) * scale)), int(round(float(y) * scale))]
                for x, y in region["polygon_ds"]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [pts], region_colors[region_index % len(region_colors)])
        region_index += 1
    vis = cv2.addWeighted(overlay, 0.14, vis, 0.86, 0)

    for name, line in reconstruction.get("target_lines", {}).items():
        if line.get("status") != "detected":
            continue
        points = [
            (int(round(float(x) * scale)), int(round(float(y) * scale)))
            for x, y in line["points_ds"]
        ]
        for p1, p2 in zip(points, points[1:]):
            cv2.line(vis, p1, p2, (15, 15, 15), display_px(9, display_scale), cv2.LINE_AA)
            cv2.line(vis, p1, p2, (0, 245, 0), display_px(4, display_scale), cv2.LINE_AA)
        if points:
            mid = points[len(points) // 2]
            label = TARGET_LINE_SHORT_NAMES.get(name, name)
            cv2.putText(vis, label, (mid[0] + display_px(8, display_scale), mid[1] - display_px(8, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.46, display_scale), (0, 0, 0), display_px(4, display_scale), cv2.LINE_AA)
            cv2.putText(vis, label, (mid[0] + display_px(8, display_scale), mid[1] - display_px(8, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.46, display_scale), (0, 245, 0), display_px(2, display_scale), cv2.LINE_AA)

    h, w = vis.shape[:2]
    for point in reconstruction.get("target_keypoints", []):
        x, y = point["point_ds"]
        px = int(round(float(x) * scale))
        py = int(round(float(y) * scale))
        color = (0, 255, 255) if point.get("status") == "observed" else (255, 255, 0)
        radius = display_px(10, display_scale)
        cv2.circle(vis, (px, py), radius + display_px(3, display_scale), (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), radius, color, -1, cv2.LINE_AA)
        label_x = min(max(px + display_px(12, display_scale), 4), max(4, w - display_px(70, display_scale)))
        label_y = min(max(py - display_px(10, display_scale), display_px(24, display_scale)), h - display_px(8, display_scale))
        cv2.putText(vis, str(point["id"]), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.58, display_scale), (0, 0, 0), display_px(5, display_scale), cv2.LINE_AA)
        cv2.putText(vis, str(point["id"]), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.58, display_scale), color, display_px(2, display_scale), cv2.LINE_AA)

    text = (
        f"halfcourt: {reconstruction['status'].upper()}  side={reconstruction['half_side']}  "
        f"kp={reconstruction['counts']['target_keypoints']}/8  "
        f"lines={reconstruction['counts']['target_lines']}  "
        f"regions={reconstruction['counts']['target_regions']}"
    )
    cv2.putText(vis, text, (display_px(16, display_scale), display_px(30, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.50, display_scale), (0, 0, 0), display_px(4, display_scale), cv2.LINE_AA)
    cv2.putText(vis, text, (display_px(16, display_scale), display_px(30, display_scale)), cv2.FONT_HERSHEY_SIMPLEX, display_font(0.50, display_scale), (255, 255, 255), display_px(2, display_scale), cv2.LINE_AA)
    return vis


def draw_full_court_calibration(
    frame: np.ndarray,
    completion: dict,
) -> np.ndarray:
    """Draw the homography-completed 14-point court on the source image."""
    vis = cv2.convertScaleAbs(frame, alpha=0.80, beta=0)
    height, width = vis.shape[:2]
    point_by_id = {
        str(point["id"]): point
        for point in completion.get("completed_keypoints_14", [])
        if point.get("homography_point") is not None
    }

    def clipped_point(point: Iterable[float]) -> tuple[int, int]:
        x, y = [float(value) for value in point]
        return (
            int(round(np.clip(x, -1_000_000.0, 1_000_000.0))),
            int(round(np.clip(y, -1_000_000.0, 1_000_000.0))),
        )

    for start_id, end_id in FULL_COURT_LINE_SEGMENTS:
        start = point_by_id.get(start_id)
        end = point_by_id.get(end_id)
        if start is None or end is None:
            continue
        p1 = clipped_point(start["homography_point"])
        p2 = clipped_point(end["homography_point"])
        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, width, height),
            p1,
            p2,
        )
        if visible:
            cv2.line(vis, clipped_start, clipped_end, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(vis, clipped_start, clipped_end, (255, 255, 0), 3, cv2.LINE_AA)

    for point_id, point in point_by_id.items():
        if not point.get("in_image", False):
            continue
        px, py = clipped_point(point["homography_point"])
        completed = point.get("status") == "homography_completed"
        color = (255, 255, 0) if completed else (0, 255, 0)
        cv2.circle(vis, (px, py), 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            point_id,
            (px + 9, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

    quality = completion.get("quality", {})
    method = str(completion.get("calibration_method", ""))
    status_label = (
        "MANUAL_ROUGH"
        if method == "manual_rough_fixed_view"
        else str(completion.get("status", "unavailable")).upper()
    )
    text = (
        f"full court calibration: {status_label}  "
        f"kp={len(point_by_id)}/14  "
        f"src={completion.get('source_keypoint_count', 0)}  "
        f"err={float(quality.get('mean_reprojection_error_px', 0.0)):.2f}px"
    )
    cv2.putText(
        vis,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def _court_to_topdown_matrix(
    pixels_per_foot: float,
    margin_px: int,
) -> np.ndarray:
    return np.asarray(
        [
            [float(pixels_per_foot), 0.0, float(margin_px)],
            [0.0, float(pixels_per_foot), float(margin_px)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _draw_topdown_court(
    image: np.ndarray,
    pixels_per_foot: float,
    margin_px: int,
) -> None:
    def canvas_point(point_id: str) -> tuple[int, int]:
        x_ft, y_ft = CANONICAL_KEYPOINTS_FT[point_id]
        return (
            int(round(float(margin_px) + float(x_ft) * pixels_per_foot)),
            int(round(float(margin_px) + float(y_ft) * pixels_per_foot)),
        )

    for start_id, end_id in FULL_COURT_LINE_SEGMENTS:
        cv2.line(
            image,
            canvas_point(start_id),
            canvas_point(end_id),
            (10, 10, 10),
            8,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            canvas_point(start_id),
            canvas_point(end_id),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )


def render_topdown_projection(
    frame: np.ndarray,
    calibration: dict,
    half_side: str,
    pixels_per_foot: float = 30.0,
    margin_px: int = 60,
) -> tuple[np.ndarray, dict]:
    """Warp one fixed-camera frame into its half of a standard 20x44 ft court."""
    court_width_px = int(round(20.0 * pixels_per_foot))
    court_length_px = int(round(44.0 * pixels_per_foot))
    canvas_width = court_width_px + 2 * int(margin_px)
    canvas_height = court_length_px + 2 * int(margin_px)
    canvas = np.full((canvas_height, canvas_width, 3), 28, dtype=np.uint8)
    image_to_court = calibration.get("homography_image_to_court")
    valid = bool(calibration.get("valid")) and image_to_court is not None
    metadata = {
        "status": "available" if valid else "unavailable",
        "coordinate_system": "pickleball_full_court_ft",
        "coordinate_system_version": 1,
        "canvas_size_px": [canvas_width, canvas_height],
        "pixels_per_foot": float(pixels_per_foot),
        "margin_px": int(margin_px),
        "source_half_side": half_side,
        "court_bounds_ft": [0.0, 0.0, 20.0, 44.0],
        "source_half_bounds_ft": (
            [0.0, 0.0, 20.0, 22.0]
            if half_side == "left"
            else [0.0, 22.0, 20.0, 44.0]
        ),
        "image_to_topdown": None,
    }
    if valid:
        court_to_canvas = _court_to_topdown_matrix(
            pixels_per_foot,
            margin_px,
        )
        image_to_topdown = (
            court_to_canvas
            @ np.asarray(image_to_court, dtype=np.float64)
        )
        metadata["image_to_topdown"] = image_to_topdown.astype(float).tolist()

    _draw_topdown_court(canvas, pixels_per_foot, margin_px)
    cv2.putText(
        canvas,
        f"{half_side.upper()} COURT FRAMEWORK ({metadata['status'].upper()})",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas, metadata


def stitch_topdown_projections(
    left_projection: np.ndarray,
    right_projection: np.ndarray,
    pixels_per_foot: float = 30.0,
    margin_px: int = 60,
) -> np.ndarray:
    """Join two rectified half-court images at the canonical net."""
    if left_projection.shape != right_projection.shape:
        raise ValueError("left/right top-down projection shapes do not match")
    stitched = np.full_like(left_projection, 28)
    net_y = int(round(float(margin_px) + 22.0 * pixels_per_foot))
    court_bottom = int(round(float(margin_px) + 44.0 * pixels_per_foot))
    stitched[:net_y + 1] = left_projection[:net_y + 1]
    stitched[net_y:court_bottom + 1] = right_projection[net_y:court_bottom + 1]
    stitched[:int(margin_px)] = 28
    _draw_topdown_court(stitched, pixels_per_foot, margin_px)
    cv2.putText(
        stitched,
        "STITCHED FIXED-VIEW COURT  20 x 44 ft",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return stitched


# Pickleball half-court topology and semantic reconstruction.

LEFT_KEYPOINTS = {
    ("top", "baseline"): "kp2",
    ("top", "kitchen"): "kp5",
    ("top", "net"): "kp7",
    ("center", "baseline"): "kp1",
    ("center", "kitchen"): "kp4",
    ("bottom", "baseline"): "kp0",
    ("bottom", "kitchen"): "kp3",
    ("bottom", "net"): "kp6",
}


RIGHT_KEYPOINTS = {
    ("top", "baseline"): "kp13",
    ("top", "kitchen"): "kp10",
    ("top", "net"): "kp7",
    ("center", "baseline"): "kp12",
    ("center", "kitchen"): "kp9",
    ("bottom", "baseline"): "kp11",
    ("bottom", "kitchen"): "kp8",
    ("bottom", "net"): "kp6",
}


HALF_LINE_KEYPOINTS = {
    "left": {
        "top_sideline": ["kp2", "kp5", "kp7"],
        "bottom_sideline": ["kp0", "kp3", "kp6"],
        "left_baseline": ["kp0", "kp1", "kp2"],
        "left_non_volley_line": ["kp3", "kp4", "kp5"],
        "left_centerline": ["kp1", "kp4"],
        "net_line": ["kp6", "kp7"],
    },
    "right": {
        "top_sideline": ["kp7", "kp10", "kp13"],
        "bottom_sideline": ["kp6", "kp8", "kp11"],
        "net_line": ["kp6", "kp7"],
        "right_non_volley_line": ["kp8", "kp9", "kp10"],
        "right_baseline": ["kp11", "kp12", "kp13"],
        "right_centerline": ["kp9", "kp12"],
    },
}


HALF_REGION_KEYPOINTS = {
    "left": {
        "left_upper_service_area": ["kp1", "kp2", "kp5", "kp4"],
        "left_lower_service_area": ["kp0", "kp1", "kp4", "kp3"],
        "left_non_volley_zone": ["kp3", "kp5", "kp7", "kp6"],
    },
    "right": {
        "right_non_volley_zone": ["kp6", "kp7", "kp10", "kp8"],
        "right_upper_service_area": ["kp9", "kp10", "kp13", "kp12"],
        "right_lower_service_area": ["kp8", "kp9", "kp12", "kp11"],
    },
}


def transform_point(
    point_xy: Iterable[float],
    homography: Iterable[Iterable[float]],
) -> list[float] | None:
    """Apply a 3x3 homography to one point and return plain JSON-safe floats."""
    point = np.asarray(list(point_xy), dtype=np.float64)
    matrix = np.asarray(list(homography), dtype=np.float64)
    if point.shape != (2,) or matrix.shape != (3, 3):
        return None
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(matrix)):
        return None
    projected = matrix @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if not np.all(np.isfinite(projected)) or abs(float(projected[2])) < 1e-10:
        return None
    return [
        float(projected[0] / projected[2]),
        float(projected[1] / projected[2]),
    ]


def complete_full_court_calibration(
    reconstruction: dict,
    image_shape: tuple[int, int],
    ransac_reproj_threshold_px: float = 8.0,
    max_mean_reproj_error_px: float = 12.0,
    min_inlier_ratio: float = 0.50,
) -> dict:
    """Fit court→image homography and complete all canonical kp0..kp13.

    The half-court detector remains responsible for semantic point discovery.
    This function consumes those points, estimates one fixed-camera planar
    homography, and projects every canonical court keypoint back to the image.
    Points outside the image are retained and marked ``in_image=false``.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    warnings: list[str] = [
        "ground_plane_homography_only",
        "airborne_ball_projection_is_line_of_sight_ground_plane_approximation",
    ]
    source_by_id: dict[str, dict] = {}
    for point in reconstruction.get("target_keypoints", []):
        point_id = str(point.get("id", ""))
        image_xy = point.get("point")
        if point_id not in CANONICAL_KEYPOINTS_FT:
            continue
        if not isinstance(image_xy, (list, tuple)) or len(image_xy) < 2:
            continue
        values = np.asarray(image_xy[:2], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            continue
        source_by_id[point_id] = {
            **point,
            "point": [float(values[0]), float(values[1])],
        }

    source_ids = sorted(
        source_by_id,
        key=lambda value: int(value.removeprefix("kp")),
    )
    unavailable = {
        "status": "unavailable",
        "valid": False,
        "coordinate_system": "pickleball_full_court_ft",
        "coordinate_system_version": 1,
        "unit": "ft",
        "source_half_side": reconstruction.get("half_side"),
        "source_keypoint_ids": source_ids,
        "source_keypoint_count": len(source_ids),
        "homography_court_to_image": None,
        "homography_image_to_court": None,
        "quality": {},
        "completed_keypoints_14": [],
        "warnings": warnings,
    }
    if len(source_ids) < 4:
        unavailable["warnings"] = [*warnings, f"too_few_semantic_points:{len(source_ids)}"]
        return unavailable

    court_points = np.asarray(
        [CANONICAL_KEYPOINTS_FT[point_id] for point_id in source_ids],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [source_by_id[point_id]["point"] for point_id in source_ids],
        dtype=np.float64,
    )
    court_area = float(cv2.contourArea(cv2.convexHull(court_points.astype(np.float32))))
    image_area = float(cv2.contourArea(cv2.convexHull(image_points.astype(np.float32))))
    if court_area <= 1e-6 or image_area <= 1e-3:
        unavailable["status"] = "invalid"
        unavailable["quality"] = {
            "court_hull_area_ft2": court_area,
            "image_hull_area_px2": image_area,
        }
        unavailable["warnings"] = [*warnings, "degenerate_semantic_points"]
        return unavailable

    method = cv2.RANSAC if len(source_ids) > 4 else 0
    try:
        court_to_image, mask = cv2.findHomography(
            court_points,
            image_points,
            method,
            float(ransac_reproj_threshold_px),
        )
    except cv2.error:
        court_to_image, mask = None, None
    if court_to_image is None or not np.all(np.isfinite(court_to_image)):
        unavailable["status"] = "invalid"
        unavailable["warnings"] = [*warnings, "homography_estimation_failed"]
        return unavailable

    scale = float(court_to_image[2, 2])
    if abs(scale) > 1e-10:
        court_to_image = court_to_image / scale
    try:
        image_to_court = np.linalg.inv(court_to_image)
    except np.linalg.LinAlgError:
        unavailable["status"] = "invalid"
        unavailable["warnings"] = [*warnings, "homography_not_invertible"]
        return unavailable
    if not np.all(np.isfinite(image_to_court)):
        unavailable["status"] = "invalid"
        unavailable["warnings"] = [*warnings, "homography_inverse_not_finite"]
        return unavailable

    reprojected = cv2.perspectiveTransform(
        court_points.astype(np.float32).reshape(-1, 1, 2),
        court_to_image,
    ).reshape(-1, 2)
    errors = np.linalg.norm(reprojected.astype(np.float64) - image_points, axis=1)
    if mask is None:
        inlier_mask = np.ones(len(source_ids), dtype=bool)
    else:
        inlier_mask = mask.reshape(-1).astype(bool)
    inlier_ratio = float(np.mean(inlier_mask)) if len(inlier_mask) else 0.0
    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    p95_error = float(np.percentile(errors, 95))
    condition_number = float(np.linalg.cond(court_to_image))

    if len(source_ids) == 4:
        warnings.append("minimum_four_point_fit_has_no_redundant_validation")
    if inlier_ratio < float(min_inlier_ratio):
        warnings.append(f"low_inlier_ratio:{inlier_ratio:.3f}")
    if mean_error > float(max_mean_reproj_error_px):
        warnings.append(f"high_mean_reprojection_error_px:{mean_error:.3f}")
    if not np.isfinite(condition_number) or condition_number > 1e12:
        warnings.append(f"ill_conditioned_homography:{condition_number:.3e}")

    valid = (
        inlier_ratio >= float(min_inlier_ratio)
        and mean_error <= float(max_mean_reproj_error_px)
        and np.isfinite(condition_number)
        and condition_number <= 1e12
    )
    all_ids = sorted(
        CANONICAL_KEYPOINTS_FT,
        key=lambda value: int(value.removeprefix("kp")),
    )
    completed = []
    error_by_id = {
        point_id: float(error)
        for point_id, error in zip(source_ids, errors)
    }
    inlier_by_id = {
        point_id: bool(value)
        for point_id, value in zip(source_ids, inlier_mask)
    }
    for point_id in all_ids:
        court_xy = CANONICAL_KEYPOINTS_FT[point_id]
        homography_point = transform_point(court_xy, court_to_image)
        if homography_point is None:
            warnings.append(f"completion_failed:{point_id}")
            continue
        source = source_by_id.get(point_id)
        raw_point = source["point"] if source is not None else homography_point
        in_image = (
            0.0 <= float(homography_point[0]) < float(width)
            and 0.0 <= float(homography_point[1]) < float(height)
        )
        completed.append({
            "id": point_id,
            "court_xy_ft": list(court_xy),
            "point": [float(raw_point[0]), float(raw_point[1])],
            "homography_point": [
                float(homography_point[0]),
                float(homography_point[1]),
            ],
            "status": (
                str(source.get("status", "detected"))
                if source is not None
                else "homography_completed"
            ),
            "source": (
                str(source.get("source", "halfcourt_reconstruction"))
                if source is not None
                else "homography_completed"
            ),
            "used_for_homography": source is not None,
            "homography_inlier": (
                inlier_by_id.get(point_id)
                if source is not None
                else None
            ),
            "reprojection_error_px": (
                error_by_id.get(point_id)
                if source is not None
                else None
            ),
            "in_image": bool(in_image),
        })

    if len(completed) != len(CANONICAL_KEYPOINTS_FT):
        valid = False
        warnings.append(f"incomplete_full_court:{len(completed)}/14")

    return {
        "status": "homography" if valid else "invalid",
        "valid": bool(valid),
        "coordinate_system": "pickleball_full_court_ft",
        "coordinate_system_version": 1,
        "unit": "ft",
        "source_half_side": reconstruction.get("half_side"),
        "source_keypoint_ids": source_ids,
        "source_keypoint_count": len(source_ids),
        "homography_court_to_image": court_to_image.astype(float).tolist(),
        "homography_image_to_court": image_to_court.astype(float).tolist(),
        "quality": {
            "ransac_reprojection_threshold_px": float(ransac_reproj_threshold_px),
            "inlier_ratio": inlier_ratio,
            "mean_reprojection_error_px": mean_error,
            "max_reprojection_error_px": max_error,
            "p95_reprojection_error_px": p95_error,
            "court_hull_area_ft2": court_area,
            "image_hull_area_px2": image_area,
            "condition_number": condition_number,
            "per_point": [
                {
                    "id": point_id,
                    "reprojection_error_px": error_by_id[point_id],
                    "is_inlier": inlier_by_id[point_id],
                }
                for point_id in source_ids
            ],
        },
        "completed_keypoints_14": completed,
        "warnings": sorted(set(warnings)),
    }


def complete_from_rough_manual_keypoints(
    reconstruction: dict,
    image_shape: tuple[int, int],
    calibration_spec: dict,
) -> dict:
    """Build an explicitly low-precision fixed-view calibration from four points."""
    expected_size = calibration_spec.get("image_size")
    actual_size = [int(image_shape[1]), int(image_shape[0])]
    if expected_size != actual_size:
        raise ValueError(
            "rough calibration image_size mismatch: "
            f"expected={expected_size}, actual={actual_size}"
        )
    manual_reconstruction = {
        "half_side": reconstruction.get("half_side"),
        "target_keypoints": [
            {
                "id": point_id,
                "status": "manual_rough",
                "source": "manual_rough_fixed_view",
                "point": point_xy,
            }
            for point_id, point_xy in calibration_spec.get("keypoints", {}).items()
        ],
    }
    calibration = complete_full_court_calibration(
        manual_reconstruction,
        image_shape,
    )
    calibration["calibration_id"] = calibration_spec.get("calibration_id")
    calibration["calibration_method"] = "manual_rough_fixed_view"
    calibration["warnings"].append("auxiliary_low_precision_calibration")
    return calibration


def _angle_gap(a: float, b: float) -> float:
    delta = abs((float(a) - float(b)) % 180.0)
    return min(delta, 180.0 - delta)


def _line_geometry(line: dict) -> tuple[np.ndarray, np.ndarray, float]:
    p1 = np.asarray([float(line["x1_ds"]), float(line["y1_ds"])], dtype=np.float64)
    p2 = np.asarray([float(line["x2_ds"]), float(line["y2_ds"])], dtype=np.float64)
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
        length = 1.0
    else:
        direction /= length
    return p1, direction, length


def _point_line_distance(point: Iterable[float], line: dict) -> float:
    p0, direction, _length = _line_geometry(line)
    delta = np.asarray(point, dtype=np.float64) - p0
    return abs(float(direction[0] * delta[1] - direction[1] * delta[0]))


def _point_overrun(point: Iterable[float], line: dict) -> float:
    p0, direction, length = _line_geometry(line)
    t = float(np.dot(np.asarray(point, dtype=np.float64) - p0, direction))
    return max(-t, t - length, 0.0)


def _intersection(line_a: dict, line_b: dict) -> tuple[float, float] | None:
    pa, va, _len_a = _line_geometry(line_a)
    pb, vb, _len_b = _line_geometry(line_b)
    cross = float(va[0] * vb[1] - va[1] * vb[0])
    if abs(cross) < 1e-5:
        return None
    delta = pb - pa
    t = float((delta[0] * vb[1] - delta[1] * vb[0]) / cross)
    point = pa + t * va
    return float(point[0]), float(point[1])


def _line_quality(line: dict) -> float:
    span = float(line.get("span_ds", line.get("span", 0.0)))
    support = float(line.get("line_support", 0.0))
    return span * (0.55 + min(max(support, 0.0), 1.0))


def _selected_point_records(selection: dict) -> list[dict]:
    records = []
    for item in selection["selected"]:
        cluster = item["cluster"]
        records.append(
            {
                "id": int(item["id"]),
                "point_ds": [float(value) for value in item["point"]],
                "line_ids": {int(value) for value in cluster["line_ids"]},
                "best_line_ids": {int(value) for value in cluster["best_line_ids"]},
                "observed": int(cluster["observed_members"]) > 0,
            }
        )
    return records


def _candidate_source_ids(
    selected: list[dict],
    clusters: list[dict],
    rays: list[dict],
    net: dict | None,
) -> tuple[set[int], set[int]]:
    seed_ids = {
        line_id
        for point in selected
        for line_id in point["best_line_ids"]
    }
    seed_ids.update(int(ray["source_line_id"]) for ray in rays)

    expanded = set(seed_ids)
    for cluster in clusters:
        best_ids = {int(value) for value in cluster["best_line_ids"]}
        if best_ids & seed_ids:
            expanded.update(best_ids)
    return seed_ids, expanded


def _make_source_candidates(
    selected: list[dict],
    clusters: list[dict],
    ground_lines: list[dict],
    rays: list[dict],
    net: dict | None,
    frame_shape: tuple[int, int],
) -> list[dict]:
    h, w = frame_shape
    line_by_id = {int(line["chain_id"]): line for line in ground_lines}
    seed_ids, expanded_ids = _candidate_source_ids(selected, clusters, rays, net)
    ray_ids = {int(ray["source_line_id"]) for ray in rays}
    min_span = max(70.0, min(h, w) * 0.055)
    ray_attach_tol = max(16.0, min(h, w) * 0.010)
    candidates = []

    for source_id in expanded_ids:
        if net is not None and source_id == int(net["net_line_id"]):
            continue
        source = line_by_id.get(source_id)
        if source is None:
            continue
        span = float(source.get("span_ds", 0.0))
        if source_id not in seed_ids and span < min_span:
            continue
        direct = [
            point["id"]
            for point in selected
            if source_id in point["best_line_ids"]
        ]
        attached = list(direct)
        if source_id in ray_ids:
            attached.extend(
                point["id"]
                for point in selected
                if _point_line_distance(point["point_ds"], source) <= ray_attach_tol
                and _point_overrun(point["point_ds"], source) <= max(
                    40.0,
                    float(source.get("span_ds", 0.0)) * 0.55,
                )
            )
        candidate = {
            **source,
            "id": f"S{source_id}",
            "source_line_ids": [int(source_id)],
            "attached_point_ids": sorted(set(attached)),
            "direct_point_ids": sorted(set(direct)),
            "is_seed": source_id in seed_ids,
            "is_net": False,
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda line: (
            bool(line["is_net"]),
            len(line["attached_point_ids"]),
            len(line["direct_point_ids"]),
            _line_quality(line),
        ),
        reverse=True,
    )
    return candidates


def _same_physical_line(line_a: dict, line_b: dict, frame_shape: tuple[int, int]) -> bool:
    h, w = frame_shape
    if _angle_gap(float(line_a["angle"]), float(line_b["angle"])) > 4.5:
        return False
    shared = set(line_a["attached_point_ids"]) & set(line_b["attached_point_ids"])
    distance_tol = max(7.0, min(h, w) * 0.0045)
    endpoints_a = (
        (float(line_a["x1_ds"]), float(line_a["y1_ds"])),
        (float(line_a["x2_ds"]), float(line_a["y2_ds"])),
    )
    endpoints_b = (
        (float(line_b["x1_ds"]), float(line_b["y1_ds"])),
        (float(line_b["x2_ds"]), float(line_b["y2_ds"])),
    )
    distance = min(
        min(_point_line_distance(point, line_b) for point in endpoints_a),
        min(_point_line_distance(point, line_a) for point in endpoints_b),
    )
    return distance <= distance_tol and (bool(shared) or distance <= distance_tol * 0.55)


def _merge_duplicate_candidates(
    candidates: list[dict],
    selected: list[dict],
    frame_shape: tuple[int, int],
) -> list[dict]:
    selected_by_id = {int(point["id"]): point for point in selected}

    def supported_bridge(line_a: dict, line_b: dict) -> bool:
        if _angle_gap(float(line_a["angle"]), float(line_b["angle"])) > 5.5:
            return False
        for point_a_id in line_a["attached_point_ids"]:
            for point_b_id in line_b["attached_point_ids"]:
                if int(point_a_id) == int(point_b_id):
                    continue
                point_a = np.asarray(selected_by_id[int(point_a_id)]["point_ds"], dtype=np.float64)
                point_b = np.asarray(selected_by_id[int(point_b_id)]["point_ds"], dtype=np.float64)
                delta = point_b - point_a
                distance = float(np.linalg.norm(delta))
                if distance < max(80.0, min(frame_shape) * 0.07):
                    continue
                bridge_angle = float(np.degrees(np.arctan2(delta[1], delta[0]))) % 180.0
                if (
                    _angle_gap(bridge_angle, float(line_a["angle"])) <= 5.5
                    and _angle_gap(bridge_angle, float(line_b["angle"])) <= 5.5
                ):
                    return True
        return False

    groups: list[list[dict]] = []
    for candidate in candidates:
        for group in groups:
            if _same_physical_line(group[0], candidate, frame_shape) or any(
                supported_bridge(old, candidate)
                for old in group
            ):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    merged = []
    for group_index, group in enumerate(groups):
        representative = max(group, key=_line_quality)
        attached_ids = sorted(
            {
                int(point_id)
                for line in group
                for point_id in line["attached_point_ids"]
            }
        )
        direct_ids = sorted(
            {
                int(point_id)
                for line in group
                for point_id in line["direct_point_ids"]
            }
        )
        x1 = float(representative["x1_ds"])
        y1 = float(representative["y1_ds"])
        x2 = float(representative["x2_ds"])
        y2 = float(representative["y2_ds"])
        angle = float(representative["angle"])

        fit_points = [
            selected_by_id[point_id]["point_ds"]
            for point_id in attached_ids
            if point_id in selected_by_id
        ]
        if len(fit_points) >= 2:
            points = np.asarray(fit_points, dtype=np.float64)
            center = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points - center, full_matrices=False)
            direction = vh[0]
            if direction[0] < 0:
                direction = -direction
            projection = (points - center) @ direction
            pad = max(20.0, float(representative.get("span_ds", 0.0)) * 0.12)
            p1 = center + direction * (float(projection.min()) - pad)
            p2 = center + direction * (float(projection.max()) + pad)
            x1, y1 = float(p1[0]), float(p1[1])
            x2, y2 = float(p2[0]), float(p2[1])
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
        elif len(fit_points) == 1:
            point = np.asarray(fit_points[0], dtype=np.float64)
            direction = np.asarray([x2 - x1, y2 - y1], dtype=np.float64)
            length = float(np.linalg.norm(direction))
            if length > 1e-6:
                direction /= length
                half_span = max(20.0, float(representative.get("span_ds", 0.0)) * 0.62)
                p1 = point - direction * half_span
                p2 = point + direction * half_span
                x1, y1 = float(p1[0]), float(p1[1])
                x2, y2 = float(p2[0]), float(p2[1])

        merged.append(
            {
                **representative,
                "id": f"L{group_index}",
                "x1_ds": x1,
                "y1_ds": y1,
                "x2_ds": x2,
                "y2_ds": y2,
                "angle": angle,
                "source_line_ids": sorted(
                    {
                        int(source_id)
                        for line in group
                        for source_id in line["source_line_ids"]
                    }
                ),
                "attached_point_ids": attached_ids,
                "direct_point_ids": direct_ids,
                "is_seed": any(bool(line["is_seed"]) for line in group),
                "is_net": any(bool(line["is_net"]) for line in group),
                "quality": max(_line_quality(line) for line in group),
            }
        )
    return merged


def _assign_direction_families(
    lines: list[dict],
    rays: list[dict],
    selected: list[dict],
    half_side: str,
) -> tuple[list[dict], list[dict]]:
    if len(rays) < 2:
        ranked = sorted(lines, key=lambda line: (len(line["attached_point_ids"]), line["quality"]), reverse=True)
        if len(ranked) < 2:
            return ranked, []
        prototypes = [float(ranked[0]["angle"]), float(ranked[1]["angle"])]
    else:
        prototypes = [float(rays[0]["source_angle"]), float(rays[1]["source_angle"])]

    source_to_line_id = {
        int(source_id): str(line["id"])
        for line in lines
        for source_id in line["source_line_ids"]
    }
    line_by_id = {str(line["id"]): line for line in lines}
    parent = {line_id: line_id for line_id in line_by_id}
    parity = {line_id: 0 for line_id in line_by_id}

    def find(line_id: str) -> tuple[str, int]:
        if parent[line_id] == line_id:
            return line_id, 0
        root, parent_parity = find(parent[line_id])
        parity[line_id] ^= parent_parity
        parent[line_id] = root
        return root, parity[line_id]

    def union(line_a_id: str, line_b_id: str, relation: int) -> bool:
        root_a, parity_a = find(line_a_id)
        root_b, parity_b = find(line_b_id)
        if root_a == root_b:
            return (parity_a ^ parity_b) == relation
        parent[root_b] = root_a
        parity[root_b] = parity_a ^ parity_b ^ relation
        return True

    constraints: list[tuple[int, str, str, int]] = []

    def add_constraint(priority: int, source_a: int, source_b: int, relation: int) -> None:
        line_a_id = source_to_line_id.get(int(source_a))
        line_b_id = source_to_line_id.get(int(source_b))
        if line_a_id is None or line_b_id is None or line_a_id == line_b_id:
            return
        constraints.append((priority, line_a_id, line_b_id, relation))

    if len(rays) >= 2:
        add_constraint(
            100,
            int(rays[0]["source_line_id"]),
            int(rays[1]["source_line_id"]),
            1,
        )

    ray_by_side = {str(ray["side"]): ray for ray in rays}
    baseline_ray = ray_by_side.get("left_arm" if half_side == "left" else "right_arm")
    for point in selected:
        best_ids = sorted(int(value) for value in point["best_line_ids"])
        if len(best_ids) >= 2:
            add_constraint(80 if point["observed"] else 40, best_ids[0], best_ids[1], 1)

    for _priority, line_a_id, line_b_id, relation in sorted(constraints, reverse=True):
        union(line_a_id, line_b_id, relation)

    components: dict[str, list[tuple[str, int]]] = {}
    for line_id in line_by_id:
        root, line_parity = find(line_id)
        components.setdefault(root, []).append((line_id, line_parity))

    anchor_colors: dict[str, int] = {}
    if len(rays) >= 2:
        anchor_a = source_to_line_id.get(int(rays[0]["source_line_id"]))
        anchor_b = source_to_line_id.get(int(rays[1]["source_line_id"]))
        if anchor_a is not None:
            anchor_colors[anchor_a] = 0
        if anchor_b is not None:
            anchor_colors[anchor_b] = 1

    colors: dict[str, int] = {}
    for component in components.values():
        fixed_offsets = []
        for line_id, line_parity in component:
            if line_id in anchor_colors:
                fixed_offsets.append(anchor_colors[line_id] ^ line_parity)
        if fixed_offsets:
            root_color = fixed_offsets[0]
        else:
            normal_cost = sum(
                _angle_gap(float(line_by_id[line_id]["angle"]), prototypes[line_parity])
                for line_id, line_parity in component
            )
            flipped_cost = sum(
                _angle_gap(float(line_by_id[line_id]["angle"]), prototypes[1 - line_parity])
                for line_id, line_parity in component
            )
            root_color = 1 if flipped_cost < normal_cost else 0
        for line_id, line_parity in component:
            colors[line_id] = root_color ^ line_parity

    family_a = []
    family_b = []
    for line in lines:
        color = colors.get(str(line["id"]), 0)
        family_gap = _angle_gap(float(line["angle"]), prototypes[color])
        item = {**line, "family_gap": float(family_gap)}
        if color == 0:
            item["family"] = "A"
            family_a.append(item)
        else:
            item["family"] = "B"
            family_b.append(item)

    for family in (family_a, family_b):
        family.sort(
            key=lambda line: (
                bool(line["is_net"]),
                len(line["attached_point_ids"]),
                len(line["direct_point_ids"]),
                -float(line["family_gap"]),
                float(line["quality"]),
            ),
            reverse=True,
        )
    return family_a[:8], family_b[:8]


def _nearest_point(
    point: tuple[float, float],
    records: list[dict],
    tolerance: float,
    used_ids: set[str] | None = None,
) -> dict | None:
    best = None
    best_distance = tolerance
    for record in records:
        record_id = str(record["id"])
        if used_ids is not None and record_id in used_ids:
            continue
        px, py = record["point_ds"]
        distance = float(np.hypot(point[0] - float(px), point[1] - float(py)))
        if distance <= best_distance:
            best = record
            best_distance = distance
    if best is None:
        return None
    return {**best, "distance_ds": float(best_distance)}


def _cluster_records(clusters: list[dict]) -> list[dict]:
    return [
        {
            "id": int(index),
            "point_ds": [float(value) for value in cluster["point"]],
            "observed": int(cluster["observed_members"]) > 0,
            "line_ids": [int(value) for value in cluster["line_ids"]],
        }
        for index, cluster in enumerate(clusters)
    ]


def _evaluate_grid(
    lines_a: tuple[dict, ...],
    lines_b: tuple[dict, ...],
    selected: list[dict],
    clusters: list[dict],
    frame_shape: tuple[int, int],
) -> dict | None:
    h, w = frame_shape
    margin_x = w * 0.12
    margin_y = h * 0.12
    snap_tol = max(24.0, min(h, w) * 0.018)
    selected_records = [
        {
            "id": point["id"],
            "point_ds": point["point_ds"],
            "observed": point["observed"],
        }
        for point in selected
    ]
    cluster_records = _cluster_records(clusters)
    intersections = []
    used_selected: set[str] = set()

    for line_a in lines_a:
        for line_b in lines_b:
            point = _intersection(line_a, line_b)
            if point is None:
                continue
            x, y = point
            if not (-margin_x <= x <= w + margin_x and -margin_y <= y <= h + margin_y):
                continue
            selected_match = _nearest_point(point, selected_records, snap_tol, used_selected)
            if selected_match is not None:
                used_selected.add(str(selected_match["id"]))
                final_point = point
                source = "selected"
                evidence_id = int(selected_match["id"])
                observed = bool(selected_match["observed"])
            else:
                cluster_match = _nearest_point(point, cluster_records, snap_tol)
                if cluster_match is not None:
                    final_point = point
                    source = "raw_cluster"
                    evidence_id = int(cluster_match["id"])
                    observed = bool(cluster_match["observed"])
                else:
                    final_point = point
                    source = "line_intersection"
                    evidence_id = None
                    observed = False
            intersections.append(
                {
                    "line_a_id": str(line_a["id"]),
                    "line_b_id": str(line_b["id"]),
                    "point_ds": [float(final_point[0]), float(final_point[1])],
                    "geometric_point_ds": [float(point[0]), float(point[1])],
                    "source": source,
                    "evidence_id": evidence_id,
                    "observed": observed,
                }
            )

    if len(intersections) < 4:
        return None
    selected_matches = sum(1 for point in intersections if point["source"] == "selected")
    raw_matches = sum(1 for point in intersections if point["source"] == "raw_cluster")
    observed_matches = sum(1 for point in intersections if point["observed"])
    attached_coverage = len(
        {
            point_id
            for line in (*lines_a, *lines_b)
            for point_id in line["attached_point_ids"]
        }
    )
    quality = sum(float(line["quality"]) for line in (*lines_a, *lines_b))
    supported_matches = selected_matches + raw_matches
    support_density = supported_matches / max(len(intersections), 1)
    score = (
        support_density,
        selected_matches,
        attached_coverage,
        observed_matches,
        raw_matches,
        -len(intersections),
        quality,
    )
    return {
        "family_a": list(lines_a),
        "family_b": list(lines_b),
        "intersections": intersections,
        "score": score,
    }


def _choose_grid(
    family_a: list[dict],
    family_b: list[dict],
    selected: list[dict],
    clusters: list[dict],
    frame_shape: tuple[int, int],
    rays: list[dict],
    half_side: str,
) -> dict | None:
    if len(rays) < 2:
        return None
    h, w = frame_shape
    ray_by_side = {str(ray["side"]): ray for ray in rays}
    baseline_ray = ray_by_side.get("left_arm" if half_side == "left" else "right_arm")
    bottom_ray = ray_by_side.get("right_arm" if half_side == "left" else "left_arm")
    if baseline_ray is None or bottom_ray is None:
        return None

    def line_for_source(lines: list[dict], source_id: int) -> dict | None:
        return next(
            (
                line
                for line in lines
                if int(source_id) in {int(value) for value in line["source_line_ids"]}
            ),
            None,
        )

    baseline_line = line_for_source(family_a, int(baseline_ray["source_line_id"]))
    if baseline_line is not None:
        column_lines = family_a
        row_lines = family_b
        columns_are_a = True
    else:
        baseline_line = line_for_source(family_b, int(baseline_ray["source_line_id"]))
        column_lines = family_b
        row_lines = family_a
        columns_are_a = False
    if baseline_line is None:
        return None

    bottom_line = line_for_source(row_lines, int(bottom_ray["source_line_id"]))
    if bottom_line is None:
        return None
    anchor = _intersection(baseline_line, bottom_line)
    if anchor is None:
        return None
    anchor_xy = np.asarray(anchor, dtype=np.float64)
    margin_x = w * 0.12
    margin_y = h * 0.12

    def valid_intersection(line_a: dict, line_b: dict) -> tuple[float, float] | None:
        point = _intersection(line_a, line_b)
        if point is None:
            return None
        if not (-margin_x <= point[0] <= w + margin_x and -margin_y <= point[1] <= h + margin_y):
            return None
        return point

    column_candidates = []
    for line in column_lines:
        if str(line["id"]) == str(baseline_line["id"]):
            continue
        point = valid_intersection(line, bottom_line)
        if point is None:
            continue
        distance = float(np.linalg.norm(np.asarray(point, dtype=np.float64) - anchor_xy))
        column_candidates.append(
            (
                len(line["attached_point_ids"]),
                len(line["direct_point_ids"]),
                float(line["quality"]),
                distance,
                line,
            )
        )
    if len(column_candidates) < 2:
        return None
    selected_columns = [
        item[-1]
        for item in sorted(column_candidates, reverse=True, key=lambda item: item[:-1])[:2]
    ]
    selected_columns = [baseline_line, *selected_columns]

    top_candidates = []
    for line in row_lines:
        if str(line["id"]) == str(bottom_line["id"]):
            continue
        baseline_hit = valid_intersection(line, baseline_line)
        if baseline_hit is None:
            continue
        if columns_are_a:
            provisional = _evaluate_grid(
                tuple(selected_columns),
                (bottom_line, line),
                selected,
                clusters,
                frame_shape,
            )
        else:
            provisional = _evaluate_grid(
                (bottom_line, line),
                tuple(selected_columns),
                selected,
                clusters,
                frame_shape,
            )
        if provisional is None:
            supported = 0
        else:
            supported = sum(
                1
                for point in provisional["intersections"]
                if point["source"] in {"selected", "raw_cluster"}
            )
        distance = float(
            np.linalg.norm(np.asarray(baseline_hit, dtype=np.float64) - anchor_xy)
        )
        top_candidates.append(
            (
                supported,
                len(line["attached_point_ids"]),
                len(line["direct_point_ids"]),
                distance,
                float(line["quality"]),
                line,
            )
        )
    if not top_candidates:
        return None
    top_line = max(top_candidates, key=lambda item: item[:-1])[-1]

    if columns_are_a:
        return _evaluate_grid(
            tuple(selected_columns),
            (bottom_line, top_line),
            selected,
            clusters,
            frame_shape,
        )
    return _evaluate_grid(
        (bottom_line, top_line),
        tuple(selected_columns),
        selected,
        clusters,
        frame_shape,
    )


def _intersection_record(grid: dict, row_id: str, column_id: str) -> dict | None:
    for point in grid["intersections"]:
        ids = {str(point["line_a_id"]), str(point["line_b_id"])}
        if ids == {row_id, column_id}:
            return point
    return None


def _net_line_candidate(
    inferred_net: dict | None,
    ground_lines: list[dict],
) -> dict | None:
    if inferred_net is None:
        return None
    if "line_ds" in inferred_net:
        (x1, y1), (x2, y2) = inferred_net["line_ds"]
        source_ids = []
    else:
        source_id = int(inferred_net["net_line_id"])
        source = next(
            (line for line in ground_lines if int(line["chain_id"]) == source_id),
            None,
        )
        if source is None:
            return None
        x1, y1 = float(source["x1_ds"]), float(source["y1_ds"])
        x2, y2 = float(source["x2_ds"]), float(source["y2_ds"])
        source_ids = [source_id]
    angle = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))) % 180.0
    return {
        "id": "NET",
        "x1_ds": float(x1),
        "y1_ds": float(y1),
        "x2_ds": float(x2),
        "y2_ds": float(y2),
        "angle": angle,
        "source_line_ids": source_ids,
        "attached_point_ids": [],
        "direct_point_ids": [],
        "quality": float(np.hypot(float(x2 - x1), float(y2 - y1))),
        "family": "NET",
        "is_net": True,
    }


def _segment_mask_support(
    mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    """Return the fraction of a line interval supported by white-line pixels."""
    height, width = mask.shape[:2]
    start_xy = np.asarray(start, dtype=np.float64)
    end_xy = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end_xy - start_xy))
    if length < 8.0:
        return 0.0

    sample_count = max(24, int(round(length)))
    samples = np.linspace(0.08, 0.92, sample_count)
    points = start_xy[None, :] + samples[:, None] * (end_xy - start_xy)[None, :]
    radius = max(2, int(round(min(height, width) * 0.004)))
    hits = 0
    valid = 0
    for x, y in points:
        px = int(round(float(x)))
        py = int(round(float(y)))
        if px < 0 or py < 0 or px >= width or py >= height:
            continue
        valid += 1
        x0 = max(0, px - radius)
        x1 = min(width, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(height, py + radius + 1)
        if np.any(mask[y0:y1, x0:x1] > 0):
            hits += 1
    return float(hits / valid) if valid else 0.0


def _semantic_grid_with_net(
    grid: dict,
    half_side: str,
    rays: list[dict],
    net_line: dict,
    frame_shape: tuple[int, int],
    white_centerline: np.ndarray,
) -> dict | None:
    families = [grid["family_a"], grid["family_b"]]
    row_lines = next((lines for lines in families if len(lines) == 3), None)
    column_white_lines = next((lines for lines in families if len(lines) == 2), None)
    if row_lines is None or column_white_lines is None or len(rays) < 2:
        return None

    def line_for_source(lines: list[dict], source_id: int) -> dict | None:
        return next(
            (
                line
                for line in lines
                if int(source_id) in {int(value) for value in line["source_line_ids"]}
            ),
            None,
        )

    bottom_line = None
    for ray in rays:
        source_id = int(ray["source_line_id"])
        bottom_line = bottom_line or line_for_source(row_lines, source_id)
    if bottom_line is None:
        return None

    h, w = frame_shape
    margin_x = w * 0.15
    margin_y = h * 0.15

    def distance_to_net(column: dict) -> float:
        distances = []
        for row in row_lines:
            column_hit = _intersection(row, column)
            net_hit = _intersection(row, net_line)
            if column_hit is None or net_hit is None:
                continue
            if not (
                -margin_x <= column_hit[0] <= w + margin_x
                and -margin_y <= column_hit[1] <= h + margin_y
            ):
                continue
            distances.append(float(np.linalg.norm(
                np.asarray(column_hit, dtype=np.float64)
                - np.asarray(net_hit, dtype=np.float64)
            )))
        return float(np.median(distances)) if distances else float("inf")

    ordered_columns = sorted(column_white_lines, key=distance_to_net)
    kitchen_line, baseline_line = ordered_columns
    column_net_distances = {
        str(line["id"]): distance_to_net(line)
        for line in column_white_lines
    }

    other_rows = [
        line
        for line in row_lines
        if str(line["id"]) != str(bottom_line["id"])
    ]
    if len(other_rows) != 2:
        return None

    def top_support(line: dict) -> tuple[float, int, float, int, float]:
        kitchen_hit = _intersection(line, kitchen_line)
        net_hit = _intersection(line, net_line)
        if kitchen_hit is None or net_hit is None:
            return 0.0, 0, float("-inf"), 0, float("-inf")
        inside = int(
            -margin_x <= net_hit[0] <= w + margin_x
            and -margin_y <= net_hit[1] <= h + margin_y
        )
        span = max(float(line.get("span_ds", line.get("span", 1.0))), 1.0)
        overrun_ratio = _point_overrun(net_hit, line) / span
        white_support = _segment_mask_support(
            white_centerline,
            kitchen_hit,
            net_hit,
        )
        return (
            white_support,
            inside,
            -overrun_ratio,
            len(line["attached_point_ids"]),
            float(line["quality"]),
        )

    top_line = max(other_rows, key=top_support)
    center_line = next(
        line
        for line in other_rows
        if str(line["id"]) != str(top_line["id"])
    )
    row_roles = {
        "top": top_line,
        "center": center_line,
        "bottom": bottom_line,
    }
    column_roles = {
        "baseline": baseline_line,
        "kitchen": kitchen_line,
        "net": net_line,
    }

    keypoint_map = LEFT_KEYPOINTS if half_side == "left" else RIGHT_KEYPOINTS
    keypoints = []
    for (row_role, column_role), keypoint_id in keypoint_map.items():
        row = row_roles[row_role]
        column = column_roles[column_role]
        point = _intersection(row, column)
        if point is None:
            continue
        existing = None
        if column_role != "net":
            existing = _intersection_record(
                grid,
                str(row["id"]),
                str(column["id"]),
            )
        source = "net_intersection" if column_role == "net" else "line_intersection"
        evidence_id = None
        status = "inferred"
        if existing is not None:
            source = str(existing["source"])
            evidence_id = existing["evidence_id"]
            status = "observed" if source == "selected" else "inferred"
        keypoints.append(
            {
                "id": keypoint_id,
                "status": status,
                "point_ds": [float(point[0]), float(point[1])],
                "source": source,
                "evidence_id": evidence_id,
                "row_role": row_role,
                "column_role": column_role,
                "row_line_id": str(row["id"]),
                "column_line_id": str(column["id"]),
            }
        )
    if len(keypoints) != 8:
        return None
    return {
        "row_lines": row_roles,
        "column_lines": column_roles,
        "keypoints": keypoints,
        "semantic_evidence": {
            "column_distance_to_net": column_net_distances,
            "row_white_support_kitchen_to_net": {
                str(line["id"]): float(top_support(line)[0])
                for line in other_rows
            },
        },
    }


def _point_payload(point: dict, downscale: int) -> dict:
    point_ds = [float(value) for value in point["point_ds"]]
    return {
        **point,
        "point_ds": point_ds,
        "point": [point_ds[0] * downscale, point_ds[1] * downscale],
    }


def _line_payload(line: dict, downscale: int) -> dict:
    segment_ds = [
        [float(line["x1_ds"]), float(line["y1_ds"])],
        [float(line["x2_ds"]), float(line["y2_ds"])],
    ]
    return {
        "id": str(line["id"]),
        "family": str(line.get("family", "")),
        "angle": float(line["angle"]),
        "source_line_ids": [int(value) for value in line["source_line_ids"]],
        "attached_point_ids": [int(value) for value in line["attached_point_ids"]],
        "is_net": bool(line["is_net"]),
        "segment_ds": segment_ds,
        "segment": [
            [point[0] * downscale, point[1] * downscale]
            for point in segment_ds
        ],
    }


def _build_target_lines(keypoints: list[dict], half_side: str, downscale: int) -> dict:
    point_by_id = {str(point["id"]): point for point in keypoints}
    result = {}
    for name, keypoint_ids in HALF_LINE_KEYPOINTS[half_side].items():
        present = [keypoint_id for keypoint_id in keypoint_ids if keypoint_id in point_by_id]
        if len(present) < 2:
            result[name] = {
                "status": "missing",
                "required_keypoint_ids": keypoint_ids,
                "present_keypoint_ids": present,
            }
            continue
        points_ds = [point_by_id[keypoint_id]["point_ds"] for keypoint_id in present]
        result[name] = {
            "status": "detected",
            "source": "halfcourt_grid",
            "keypoint_ids": present,
            "points_ds": points_ds,
            "points": [
                [float(point[0]) * downscale, float(point[1]) * downscale]
                for point in points_ds
            ],
        }
    return result


def _build_target_regions(keypoints: list[dict], half_side: str, downscale: int) -> dict:
    point_by_id = {str(point["id"]): point for point in keypoints}
    result = {}
    for name, keypoint_ids in HALF_REGION_KEYPOINTS[half_side].items():
        if not all(keypoint_id in point_by_id for keypoint_id in keypoint_ids):
            result[name] = {
                "status": "missing",
                "required_keypoint_ids": keypoint_ids,
            }
            continue
        polygon_ds = [point_by_id[keypoint_id]["point_ds"] for keypoint_id in keypoint_ids]
        result[name] = {
            "status": "detected",
            "source": "halfcourt_grid",
            "keypoint_ids": keypoint_ids,
            "polygon_ds": polygon_ds,
            "polygon": [
                [float(point[0]) * downscale, float(point[1]) * downscale]
                for point in polygon_ds
            ],
        }
    return result


def reconstruct_half_court(
    selection: dict,
    clusters: list[dict],
    ground_lines: list[dict],
    rays: list[dict],
    inferred_net: dict | None,
    frame_shape: tuple[int, int],
    downscale: int,
    half_side: str,
    white_centerline: np.ndarray,
) -> dict:
    selected = _selected_point_records(selection)
    source_candidates = _make_source_candidates(
        selected,
        clusters,
        ground_lines,
        rays,
        inferred_net,
        frame_shape,
    )
    structural_lines = _merge_duplicate_candidates(source_candidates, selected, frame_shape)
    family_a, family_b = _assign_direction_families(structural_lines, rays, selected, half_side)
    grid = _choose_grid(family_a, family_b, selected, clusters, frame_shape, rays, half_side)
    net_line = _net_line_candidate(inferred_net, ground_lines)
    semantic = (
        None
        if grid is None or net_line is None
        else _semantic_grid_with_net(
            grid,
            half_side,
            rays,
            net_line,
            frame_shape,
            white_centerline,
        )
    )

    if semantic is None:
        keypoints = []
        target_lines = {
            name: {"status": "missing"}
            for name in HALF_LINE_KEYPOINTS.get(half_side, {})
        }
        target_regions = {
            name: {"status": "missing"}
            for name in HALF_REGION_KEYPOINTS.get(half_side, {})
        }
        status = "failed"
        failure_reasons = ["halfcourt_grid_not_resolved"]
    else:
        keypoints = [_point_payload(point, downscale) for point in semantic["keypoints"]]
        target_lines = _build_target_lines(keypoints, half_side, downscale)
        target_regions = _build_target_regions(keypoints, half_side, downscale)
        status = "success" if len(keypoints) == 8 else "partial"
        failure_reasons = [] if status == "success" else ["halfcourt_keypoints_incomplete"]

    grid_lines = []
    grid_intersections = []
    if grid is not None:
        grid_lines = [
            _line_payload(line, downscale)
            for line in (*grid["family_a"], *grid["family_b"])
        ]
        grid_intersections = [
            _point_payload(
                {
                    "id": f"G{index}",
                    **point,
                },
                downscale,
            )
            for index, point in enumerate(grid["intersections"])
        ]

    semantic_lines = {}
    if semantic is not None:
        semantic_lines = {
            **{
                role: _line_payload(line, downscale)
                for role, line in semantic["row_lines"].items()
                if line is not None
            },
            **{
                role: _line_payload(line, downscale)
                for role, line in semantic["column_lines"].items()
            },
        }

    return {
        "status": status,
        "failure_reasons": failure_reasons,
        "half_side": half_side,
        "selected_points": [
            _point_payload(
                {
                    "id": point["id"],
                    "status": "selected",
                    "point_ds": point["point_ds"],
                    "observed": point["observed"],
                },
                downscale,
            )
            for point in selected
        ],
        "structural_lines": [
            _line_payload(line, downscale)
            for line in (*family_a, *family_b)
        ],
        "grid_lines": grid_lines,
        "grid_intersections": grid_intersections,
        "semantic_lines": semantic_lines,
        "target_keypoints": keypoints,
        "target_lines": target_lines,
        "target_regions": target_regions,
        "semantic_evidence": (
            {} if semantic is None else semantic.get("semantic_evidence", {})
        ),
        "counts": {
            "selected_points": len(selected),
            "structural_lines": len(structural_lines),
            "family_a_lines": len(family_a),
            "family_b_lines": len(family_b),
            "grid_lines": len(grid_lines),
            "grid_intersections": len(grid_intersections),
            "target_keypoints": len(keypoints),
            "target_lines": sum(
                1 for line in target_lines.values()
                if line.get("status") == "detected"
            ),
            "target_regions": sum(
                1 for region in target_regions.values()
                if region.get("status") == "detected"
            ),
        },
    }



# Public API and command-line entry point.

def _safe_output_stem(path: Path) -> str:
    return {
        "\u53f3": "right",
        "\u5de6": "left",
        "\u4fa7\u540e": "back_side",
    }.get(path.stem, path.stem.encode("ascii", "ignore").decode("ascii") or "image")


def detect_court_image(
    path: str | Path,
    output_dir: str | Path,
    downscale: int = 2,
    max_width: int = 1800,
    rough_calibration: dict | None = None,
) -> dict:
    """Detect one side-view half court and write the compact seven-stage audit."""
    path = Path(path)
    output_dir = Path(output_dir)
    if downscale < 1:
        raise ValueError("downscale must be >= 1")

    frame, input_info = read_calibration_frame(path)
    h, w = frame.shape[:2]
    if downscale > 1:
        frame_ds = cv2.resize(
            frame,
            (w // downscale, h // downscale),
            interpolation=cv2.INTER_AREA,
        )
    else:
        frame_ds = frame
    h_ds, w_ds = frame_ds.shape[:2]
    display_scale = min(1.0, max_width / float(w)) if max_width > 0 else 1.0

    full_mask_ds = np.full((h_ds, w_ds), 255, dtype=np.uint8)
    preprocess = enhance_white_lines_for_lsd(frame_ds, full_mask_ds)
    centerline = preprocess["centerline"]

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 0.55)
    raw_lines, _, _, _ = lsd.detect(centerline)
    if raw_lines is None:
        raw_lines = np.empty((0, 4), dtype=np.float32)
    segments = parse_lsd_segments(
        raw_lines.reshape(-1, 4),
        float(downscale),
        min_len_ds=max(4.0, 8.0 / downscale),
    )
    pairs, chains = build_connectivity(segments, centerline)

    refit_lines = []
    for chain_id, chain in enumerate(chains):
        line = refit_chain(chain, centerline, chain_id)
        if line is not None:
            refit_lines.append(line)
    refit_lines.sort(key=lambda item: (-item["span_ds"], -item["density"]))
    filtered = filter_refit_lines(refit_lines, (h_ds, w_ds), float(downscale))
    ground_lines = filter_ground_line_candidates(filtered, (h_ds, w_ds))

    line_candidate_mode = "sideview"
    intersection_min_y_ratio = 0.34
    selection_min_y_ratio = 0.47
    hough_lines: list[dict] = []
    direction_groups = group_lines_by_direction(ground_lines, max_groups=4)
    raw_intersections, intersection_clusters = generate_white_line_intersections(
        ground_lines,
        (h_ds, w_ds),
        min_y_ratio=intersection_min_y_ratio,
    )

    if len(intersection_clusters) < 4:
        hough_lines = build_centerline_hough_lines(centerline, float(downscale))
        merged_fallback = merge_collinear_line_candidates(
            filtered + hough_lines,
            centerline,
        )
        fallback_lines = filter_near_halfcourt_line_candidates(
            merged_fallback,
            (h_ds, w_ds),
        )
        fallback_raw, fallback_clusters = generate_white_line_intersections(
            fallback_lines,
            (h_ds, w_ds),
            min_y_ratio=0.05,
        )
        if len(fallback_clusters) > len(intersection_clusters):
            line_candidate_mode = "near_halfcourt"
            intersection_min_y_ratio = 0.05
            selection_min_y_ratio = 0.10
            ground_lines = fallback_lines
            direction_groups = group_lines_by_direction(ground_lines, max_groups=4)
            raw_intersections = fallback_raw
            intersection_clusters = fallback_clusters

    if line_candidate_mode == "near_halfcourt":
        selection_anchor = select_near_halfcourt_v_anchor(
            intersection_clusters,
            ground_lines,
            (h_ds, w_ds),
        )
    else:
        selection_anchor = select_near_v_anchor(
            intersection_clusters,
            ground_lines,
            (h_ds, w_ds),
        )
    selection_rays = (
        []
        if selection_anchor is None
        else build_v_rays_from_anchor(selection_anchor, ground_lines, (h_ds, w_ds))
    )
    selected = select_near_side_intersections(
        intersection_clusters,
        ground_lines,
        (h_ds, w_ds),
        selection_rays,
        foreground_min_y_ratio=selection_min_y_ratio,
        require_ray_membership=line_candidate_mode == "near_halfcourt",
    )

    half_side = infer_half_side(path)
    inferred_net = infer_net_midpoint_from_selected(
        selected["selected"],
        ground_lines,
        (h_ds, w_ds),
    )
    if inferred_net is None:
        inferred_net = detect_net_line_fallback(frame_ds, half_side)

    reconstruction = reconstruct_half_court(
        selected,
        intersection_clusters,
        ground_lines,
        selection_rays,
        inferred_net,
        (h_ds, w_ds),
        downscale,
        half_side,
        centerline,
    )
    if rough_calibration is None:
        calibration = complete_full_court_calibration(
            reconstruction,
            (h, w),
        )
        calibration["calibration_method"] = "automatic_semantic_keypoints"
    else:
        calibration = complete_from_rough_manual_keypoints(
            reconstruction,
            (h, w),
            rough_calibration,
        )
    reconstruction["full_court_calibration"] = calibration
    reconstruction["full_court_calibration"]["calibration_source"] = {
        "input": str(path),
        **input_info,
    }

    out = output_dir / _safe_output_stem(path)
    out.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.jpg", "*.png", "*.json"):
        for old in out.glob(pattern):
            old.unlink()

    scale = float(downscale)
    save_vis(out / "00_input.jpg", frame, max_width)
    save_vis(
        out / "01_white_centerline.jpg",
        mask_debug_overlay(
            frame,
            upscale_gray(centerline, (h, w)),
            fill_color=(0, 0, 255),
            edge_color=(0, 255, 255),
            title="white-line centerline used by LSD",
        ),
        max_width,
    )
    save_vis(
        out / "02_line_intersections.jpg",
        draw_white_line_intersections(
            frame,
            ground_lines,
            raw_intersections,
            intersection_clusters,
            scale,
            (h_ds, w_ds),
        ),
        max_width,
    )
    save_vis(
        out / "03_selected_keypoints.jpg",
        draw_selected_intersections(frame, ground_lines, selected, inferred_net, scale),
        max_width,
    )
    save_vis(
        out / "04_line_network.jpg",
        draw_reconstruction_network(frame, reconstruction, scale, display_scale),
        max_width,
    )
    save_vis(
        out / "05_completed_keypoints.jpg",
        draw_completed_grid(frame, reconstruction, scale, display_scale),
        max_width,
    )
    save_vis(
        out / "06_demo_result.jpg",
        draw_halfcourt_result(frame, reconstruction, scale, display_scale),
        max_width,
    )
    save_vis(
        out / "07_full_court_calibration.jpg",
        draw_full_court_calibration(
            frame,
            reconstruction["full_court_calibration"],
        ),
        max_width,
    )

    calibration = reconstruction["full_court_calibration"]
    topdown_projection, topdown_metadata = render_topdown_projection(
        frame,
        calibration,
        half_side,
    )
    calibration["topdown_projection"] = topdown_metadata
    save_vis(
        out / "08_topdown_projection.jpg",
        topdown_projection,
        max_width,
    )
    summary = {
        "input": str(path),
        **input_info,
        "output_dir": str(out),
        "image_shape": [h, w],
        "working_shape": [h_ds, w_ds],
        "downscale": downscale,
        "line_candidate_mode": line_candidate_mode,
        "white_threshold": float(preprocess["threshold"]),
        "raw_lsd_segments": len(segments),
        "connectable_pairs": len(pairs),
        "chains": len(chains),
        "refit_lines": len(refit_lines),
        "filtered_lines": len(filtered),
        "ground_lines": len(ground_lines),
        "hough_centerline_lines": len(hough_lines),
        "intersection_clusters": len(intersection_clusters),
        "selected_intersections": len(selected["selected"]),
        "half_side": half_side,
        "has_inferred_net": inferred_net is not None,
        "status": reconstruction["status"],
        "failure_reasons": reconstruction["failure_reasons"],
        "counts": reconstruction["counts"],
        "calibration_status": calibration["status"],
        "calibration_valid": calibration["valid"],
        "calibration_id": calibration.get("calibration_id"),
        "calibration_method": calibration.get("calibration_method"),
        "completed_keypoints_14": len(calibration["completed_keypoints_14"]),
        "calibration_mean_reprojection_error_px": calibration.get(
            "quality",
            {},
        ).get("mean_reprojection_error_px"),
        "topdown_projection_status": topdown_metadata["status"],
        "topdown_canvas_size_px": topdown_metadata["canvas_size_px"],
    }
    (out / "halfcourt_reconstruction.json").write_text(
        json.dumps(reconstruction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": out,
        "summary": summary,
        "reconstruction": reconstruction,
        "topdown_projection": topdown_projection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-contained side-view pickleball half-court detector"
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/pickleball_court_handoff")
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=1800)
    parser.add_argument(
        "--rough-calibration",
        help="JSON with fixed-view left/right manual corner approximations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rough_by_side = {}
    if args.rough_calibration:
        rough_by_side = json.loads(
            Path(args.rough_calibration).read_text(encoding="utf-8"),
        ).get("cameras", {})
    results_by_side = {}
    for item in args.inputs:
        side = infer_half_side(Path(item))
        result = detect_court_image(
            item,
            args.output_dir,
            downscale=args.downscale,
            max_width=args.max_width,
            rough_calibration=rough_by_side.get(side),
        )
        results_by_side[side] = result
        summary = result["summary"]
        display_status = (
            "rough_calibrated"
            if summary["calibration_valid"]
            and summary.get("calibration_method") == "manual_rough_fixed_view"
            else summary["status"]
        )
        print(
            f"[{display_status}] {item} -> {result['output_dir']} "
            f"keypoints={summary['counts']['target_keypoints']} "
            f"full={summary['completed_keypoints_14']}/14 "
            f"calibration={summary['calibration_status']}"
        )
    if "left" in results_by_side and "right" in results_by_side:
        pixels_per_foot = 30.0
        margin_px = 60
        stitched = stitch_topdown_projections(
            results_by_side["left"]["topdown_projection"],
            results_by_side["right"]["topdown_projection"],
            pixels_per_foot=pixels_per_foot,
            margin_px=margin_px,
        )
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        stitched_path = output_root / "09_stitched_topdown_projection.jpg"
        cv2.imwrite(str(stitched_path), stitched)
        stitched_metadata = {
            "status": "available",
            "coordinate_system": "pickleball_full_court_ft",
            "coordinate_system_version": 1,
            "court_size_ft": [20.0, 44.0],
            "net_y_ft": 22.0,
            "canvas_size_px": [int(stitched.shape[1]), int(stitched.shape[0])],
            "pixels_per_foot": pixels_per_foot,
            "margin_px": margin_px,
            "left_calibration_id": results_by_side["left"]["summary"].get(
                "calibration_id"
            ),
            "right_calibration_id": results_by_side["right"]["summary"].get(
                "calibration_id"
            ),
            "left_homography_image_to_court": results_by_side["left"][
                "reconstruction"
            ]["full_court_calibration"]["homography_image_to_court"],
            "right_homography_image_to_court": results_by_side["right"][
                "reconstruction"
            ]["full_court_calibration"]["homography_image_to_court"],
            "warnings": [
                "auxiliary_low_precision_calibration",
                "ground_plane_homography_only",
            ],
        }
        (
            output_root / "09_stitched_topdown_projection.json"
        ).write_text(
            json.dumps(stitched_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[stitched] {stitched_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
