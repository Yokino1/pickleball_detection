"""Run multi-pickleball detection and tracking on one video or a folder."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.overlay import TrackingOverlay
from src.tracking.factory import (
    build_detector,
    build_person_detector,
    build_pipeline,
    load_config,
    project_path,
)


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def discover_videos(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return supported videos in deterministic order."""
    if input_path.is_file():
        return [input_path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def batch_output_paths(input_path: Path, input_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Build collision-safe output paths while preserving input subdirectories."""
    relative = input_path.relative_to(input_root)
    output_parent = output_root / relative.parent
    return (
        output_parent / f"{input_path.stem}_tracked.mp4",
        output_parent / f"{input_path.stem}_tracking.jsonl",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-pickleball video or folder tracking")
    parser.add_argument("--config", default="configs/tracking.yaml")
    parser.add_argument("--input", help="Video file or folder; overrides input.source")
    parser.add_argument("--output", help="Single-video output path")
    parser.add_argument("--jsonl", help="Single-video JSONL output path")
    parser.add_argument(
        "--output-dir",
        default="outputs/batch_tracking",
        help="Batch output directory (default: outputs/batch_tracking)",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input subdirectories")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed batch outputs")
    parser.add_argument("--fail-fast", action="store_true", help="Stop batch after the first failed video")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def run_video(
    input_path: Path,
    output_video: Path | None,
    output_jsonl: Path | None,
    detector,
    config: dict,
    start_frame: int = 0,
    max_frames: int | None = None,
    preview: bool = False,
    person_detector=None,
) -> int:
    """Process one video while allowing the detector model to be reused."""
    output_config = config.get("output", {})
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

    person_config = config.get("runtime", {}).get("person_detection", {})
    pipeline = build_pipeline(
        config,
        fps,
        detector,
        person_detector=person_detector,
    )
    overlay = TrackingOverlay(
        trail_length=output_config.get("trail_length", 30),
        draw_raw_detections=output_config.get("draw_raw_detections", False),
        draw_players=output_config.get("draw_players", False),
    )

    writer = None
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*output_config.get("codec", "mp4v"))
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_video}")

    jsonl_handle = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = output_jsonl.open("w", encoding="utf-8")

    print(f"[input] {input_path} ({width}x{height}, {fps:.2f} fps, {total_frames} frames)")
    print(
        f"[pipeline] detector={type(detector).__name__}, "
        f"tracker={type(pipeline.tracker).__name__}"
    )
    if person_detector is not None:
        print(
            f"[person] detector={type(person_detector).__name__}, "
            f"interval={person_config.get('interval_frames', 5)} frames"
        )
    started = time.perf_counter()
    processed = 0
    frame_index = max(0, start_frame)
    try:
        while True:
            if max_frames is not None and processed >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, frame_index, frame_index / fps)
            if (
                result.diagnostics["detector_error"]
                and config.get("runtime", {}).get("fail_on_detector_error", True)
            ):
                raise RuntimeError("Detector failed; stopping instead of writing misleading output")
            if (
                result.diagnostics["person_detector_error"]
                and person_config.get("fail_on_error", True)
            ):
                error = getattr(person_detector, "last_error", "unknown error")
                raise RuntimeError(f"Person detector failed: {error}")
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            if writer is not None or preview:
                rendered = overlay.draw(frame.copy(), result)
                if writer is not None:
                    writer.write(rendered)
                if preview:
                    cv2.imshow("Pickleball Tracking", rendered)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if jsonl_handle is not None:
            jsonl_handle.close()
        if preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    throughput = processed / elapsed if elapsed else 0.0
    print(f"[done] {processed} frames in {elapsed:.2f}s ({throughput:.2f} fps)")
    if getattr(detector, "disabled", False):
        print("[warning] detector failed during processing; inspect stderr")
        return 2
    if output_video:
        print(f"[output] video: {output_video}")
    if output_jsonl:
        print(f"[output] jsonl: {output_jsonl}")
    return 0


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    input_path = project_path(args.input or config.get("input", {}).get("source", ""))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file or folder does not exist: {input_path}")

    videos = discover_videos(input_path, recursive=args.recursive)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {input_path}")

    detector = build_detector(config)
    person_detector = build_person_detector(config)
    output_config = config.get("output", {})
    if input_path.is_file():
        video_value = args.output or output_config.get("video")
        jsonl_value = args.jsonl or output_config.get("jsonl")
        output_video = project_path(video_value) if video_value else None
        output_jsonl = project_path(jsonl_value) if jsonl_value else None
        return run_video(
            input_path,
            output_video,
            output_jsonl,
            detector,
            config,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            preview=args.preview,
            person_detector=person_detector,
        )

    if args.output or args.jsonl:
        raise ValueError("Use --output-dir for folder input; --output and --jsonl are single-video options")

    output_root = project_path(args.output_dir)
    print(f"[batch] found {len(videos)} video(s) in {input_path}")
    succeeded = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    for index, video_path in enumerate(videos, start=1):
        output_video, output_jsonl = batch_output_paths(video_path, input_path, output_root)
        print(f"[batch {index}/{len(videos)}] {video_path}")
        if args.skip_existing and output_video.exists() and output_jsonl.exists():
            print("[skip] both output files already exist")
            skipped += 1
            continue
        try:
            code = run_video(
                video_path,
                output_video,
                output_jsonl,
                detector,
                config,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                preview=args.preview,
                person_detector=person_detector,
            )
            if code != 0:
                raise RuntimeError(f"processing returned exit code {code}")
            succeeded += 1
        except Exception as exc:
            failures.append((video_path, str(exc)))
            print(f"[failed] {video_path}: {exc}", file=sys.stderr)
            if (
                args.fail_fast
                or getattr(detector, "disabled", False)
                or getattr(person_detector, "disabled", False)
            ):
                break

    print(
        f"[batch-done] succeeded={succeeded} skipped={skipped} failed={len(failures)} "
        f"output={output_root}"
    )
    for video_path, message in failures:
        print(f"[batch-error] {video_path}: {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
