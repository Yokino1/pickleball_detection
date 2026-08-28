"""Measure detector plus tracker latency on the actual deployment machine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.factory import build_detector, build_tracker, project_path
from src.tracking.ball_pipeline import BallTrackingPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tracking_edge.yaml")
    parser.add_argument("--input")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", default="artifacts/benchmarks/runtime.json")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_path = project_path(args.input or config["input"]["source"])
    detector = build_detector(config)
    pipeline = BallTrackingPipeline(
        detector,
        build_tracker(config),
        detector_interval=config.get("runtime", {}).get("detector_interval", 1),
    )
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")

    latencies = []
    detector_latencies = []
    frame_index = 0
    while len(latencies) < args.frames:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        result = pipeline.process_frame(frame, frame_index)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if frame_index >= args.warmup:
            latencies.append(elapsed_ms)
            if not result.diagnostics["detector_skipped"]:
                detector_latencies.append(result.diagnostics["detector_ms"])
        frame_index += 1
    capture.release()
    if not latencies:
        raise RuntimeError("No benchmark frames were processed")

    model_path = project_path(config["detector"]["model"])
    report = {
        "config": str(config_path),
        "input": str(input_path),
        "measured_frames": len(latencies),
        "pipeline_ms_mean": float(np.mean(latencies)),
        "pipeline_ms_p50": float(np.percentile(latencies, 50)),
        "pipeline_ms_p95": float(np.percentile(latencies, 95)),
        "throughput_fps": float(1000.0 / np.mean(latencies)),
        "detector_ms_mean": float(np.mean(detector_latencies)) if detector_latencies else 0.0,
        "detector_ms_p95": float(np.percentile(detector_latencies, 95)) if detector_latencies else 0.0,
        "model_size_bytes": model_path.stat().st_size,
        "detector_disabled": getattr(detector, "disabled", False),
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if report["detector_disabled"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
