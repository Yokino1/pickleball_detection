"""Generate representative court-split preview images without modifying video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SAMPLE_RATIOS = (0.05, 0.50, 0.95)


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def _resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    scale = width / frame.shape[1]
    return cv2.resize(
        frame,
        (width, max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Could not encode preview: {path}")
    encoded.tofile(str(path))


def _annotate_split(frame: np.ndarray, split_x: int) -> np.ndarray:
    annotated = frame.copy()
    cv2.line(
        annotated,
        (split_x, 0),
        (split_x, annotated.shape[0] - 1),
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"proposed net split x={split_x}",
        (max(12, split_x - 230), 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _make_crop_pair(frame: np.ndarray, split_x: int) -> np.ndarray:
    left = frame[:, :split_x]
    right = frame[:, split_x:]
    target_width = 480
    left = _resize_width(left, target_width)
    right = _resize_width(right, target_width)
    target_height = min(left.shape[0], right.shape[0])
    left = left[:target_height]
    right = right[:target_height]
    cv2.putText(
        left,
        "LEFT HALF",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        "RIGHT HALF",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return np.hstack([left, right])


def make_preview(
    input_path: Path,
    output_dir: Path,
    split_ratio: float | None,
) -> tuple[Path, Path | None]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    middle_original: np.ndarray | None = None
    for ratio in SAMPLE_RATIOS:
        frame_index = min(frame_count - 1, int(round((frame_count - 1) * ratio)))
        frame = _read_frame(capture, frame_index)
        if ratio == 0.50:
            middle_original = frame.copy()
        frame = _resize_width(frame, 960)
        cv2.putText(
            frame,
            f"frame={frame_index}  time={frame_index / max(fps, 1.0):.2f}s",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if split_ratio is not None:
            preview_split_x = int(round(frame.shape[1] * split_ratio))
            frame = _annotate_split(frame, preview_split_x)
        frames.append(frame)
    capture.release()
    preview = np.vstack(frames)
    output_path = output_dir / f"{input_path.stem}_raw_samples.jpg"
    _write_image(output_path, preview)
    split_output_path: Path | None = None
    if split_ratio is not None and middle_original is not None:
        original_split_x = int(round(middle_original.shape[1] * split_ratio))
        full = _resize_width(
            _annotate_split(middle_original, original_split_x),
            960,
        )
        pair = _make_crop_pair(middle_original, original_split_x)
        split_preview = np.vstack([full, pair])
        split_output_path = output_dir / f"{input_path.stem}_split_preview.jpg"
        _write_image(split_output_path, split_preview)
        print(
            f"{input_path.name}: {middle_original.shape[1]}x"
            f"{middle_original.shape[0]}, fps={fps:.3f}, frames={frame_count}, "
            f"split_x={original_split_x}"
        )
    return output_path, split_output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-ratios", type=float, nargs="*")
    args = parser.parse_args()
    ratios = args.split_ratios or [None] * len(args.input)
    if len(ratios) != len(args.input):
        parser.error("--split-ratios must have one value per input video")
    for input_path, split_ratio in zip(args.input, ratios):
        print(make_preview(input_path, args.output_dir, split_ratio))


if __name__ == "__main__":
    main()
