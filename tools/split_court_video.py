"""Split a fixed-camera court video into synchronized left/right halves."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def split_video(
    input_path: Path,
    split_x: int,
    left_output: Path,
    right_output: Path,
    codec: str = "mp4v",
) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not 0 < split_x < width:
        capture.release()
        raise ValueError(f"split_x must be inside 0..{width}: {split_x}")

    left_output.parent.mkdir(parents=True, exist_ok=True)
    right_output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    left_writer = cv2.VideoWriter(
        str(left_output),
        fourcc,
        fps,
        (split_x, height),
    )
    right_writer = cv2.VideoWriter(
        str(right_output),
        fourcc,
        fps,
        (width - split_x, height),
    )
    if not left_writer.isOpened() or not right_writer.isOpened():
        capture.release()
        left_writer.release()
        right_writer.release()
        raise RuntimeError("Cannot create one or both split output videos")

    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            left_writer.write(frame[:, :split_x])
            right_writer.write(frame[:, split_x:])
            frames += 1
    finally:
        capture.release()
        left_writer.release()
        right_writer.release()

    if expected_frames > 0 and frames != expected_frames:
        raise RuntimeError(
            f"Frame count changed during split: expected={expected_frames}, wrote={frames}"
        )
    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-x", type=int, required=True)
    parser.add_argument("--left-output", type=Path, required=True)
    parser.add_argument("--right-output", type=Path, required=True)
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()

    frames, fps = split_video(
        args.input,
        args.split_x,
        args.left_output,
        args.right_output,
        args.codec,
    )
    print(f"[done] frames={frames}, fps={fps:.3f}")
    print(f"[left] {args.left_output}")
    print(f"[right] {args.right_output}")


if __name__ == "__main__":
    main()
