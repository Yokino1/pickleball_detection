"""Run multi-pickleball detection and tracking on a video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.ball_detector import NullBallDetector, YoloBallDetector
from src.tracking.ball_pipeline import BallTrackingPipeline
from src.tracking.multi_ball_tracker import MultiBallTracker
from src.tracking.onnx_detector import OnnxBallDetector
from src.tracking.overlay import TrackingOverlay


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema_version") != 1:
        raise ValueError("Config must declare schema_version: 1")
    return config


def build_detector(config: dict):
    detector_config = config.get("detector", {})
    model_value = detector_config.get("model")
    backend = str(detector_config.get("backend", "auto")).lower()
    if backend == "null" or model_value in (None, "null"):
        return NullBallDetector()

    model_path = project_path(model_value)
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model does not exist: {model_path}")
    if backend == "auto":
        backend = "onnxruntime" if model_path.suffix.lower() == ".onnx" else "ultralytics"

    common = {
        "model_path": model_path,
        "ball_class_id": detector_config.get("ball_class_id", 0),
        "conf_threshold": detector_config.get("low_conf", 0.08),
        "iou_threshold": detector_config.get("iou_threshold", 0.5),
        "imgsz": detector_config.get("imgsz", 640),
        "max_detections": detector_config.get("max_detections", 32),
        "exclude_region": detector_config.get("exclude_region"),
    }
    if backend == "onnxruntime":
        return OnnxBallDetector(providers=detector_config.get("providers"), **common)
    if backend == "ultralytics":
        return YoloBallDetector(device=detector_config.get("device"), **common)
    raise ValueError(f"Unsupported detector backend: {backend}")


def build_tracker(config: dict) -> MultiBallTracker:
    values = dict(config.get("tracker", {}))
    return MultiBallTracker(**values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-pickleball video tracking")
    parser.add_argument("--config", default="configs/tracking.yaml")
    parser.add_argument("--input", help="Override input.source")
    parser.add_argument("--output", help="Override output.video")
    parser.add_argument("--jsonl", help="Override output.jsonl")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    input_path = project_path(args.input or config.get("input", {}).get("source", ""))
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_config = config.get("output", {})
    video_value = args.output or output_config.get("video")
    jsonl_value = args.jsonl or output_config.get("jsonl")
    output_video = project_path(video_value) if video_value else None
    output_jsonl = project_path(jsonl_value) if jsonl_value else None

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))

    detector = build_detector(config)
    tracker = build_tracker(config)
    pipeline = BallTrackingPipeline(
        detector,
        tracker,
        detector_interval=config.get("runtime", {}).get("detector_interval", 1),
    )
    overlay = TrackingOverlay(
        trail_length=output_config.get("trail_length", 30),
        draw_raw_detections=output_config.get("draw_raw_detections", False),
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
    print(f"[pipeline] detector={type(detector).__name__}, tracker={type(tracker).__name__}")
    started = time.perf_counter()
    processed = 0
    frame_index = max(0, args.start_frame)
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
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
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            if writer is not None or args.preview:
                rendered = overlay.draw(frame.copy(), result)
                if writer is not None:
                    writer.write(rendered)
                if args.preview:
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
        if args.preview:
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


if __name__ == "__main__":
    raise SystemExit(main())
