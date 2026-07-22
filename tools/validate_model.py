"""Validate a PT/ONNX model and optionally enforce quantization tolerances."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def metric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="datasets/cleaned_ball_detection/data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="Baseline JSON from this tool")
    parser.add_argument("--max-map50-drop", type=float, default=0.015)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = resolve_path(args.model)
    results = YOLO(str(model_path)).val(
        data=str(resolve_path(args.data)),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
    )
    box = results.box
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "data": str(resolve_path(args.data)),
        "split": args.split,
        "imgsz": args.imgsz,
        "map50_95": metric(getattr(box, "map", 0.0)),
        "map50": metric(getattr(box, "map50", 0.0)),
        "precision": metric(getattr(box, "mp", 0.0)),
        "recall": metric(getattr(box, "mr", 0.0)),
        "accepted": True,
        "acceptance_failures": [],
    }
    if args.baseline:
        baseline = json.loads(resolve_path(args.baseline).read_text(encoding="utf-8"))
        map50_drop = baseline["map50"] - report["map50"]
        recall_drop = baseline["recall"] - report["recall"]
        report["baseline"] = str(resolve_path(args.baseline))
        report["map50_drop"] = map50_drop
        report["recall_drop"] = recall_drop
        if map50_drop > args.max_map50_drop:
            report["acceptance_failures"].append("map50_drop")
        if recall_drop > args.max_recall_drop:
            report["acceptance_failures"].append("recall_drop")
        report["accepted"] = not report["acceptance_failures"]

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
