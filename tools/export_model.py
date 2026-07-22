"""Export a trained detector and record reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--data", default="datasets/cleaned_ball_detection/data.yaml")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--nms", action="store_true", default=True)
    parser.add_argument("--no-nms", dest="nms", action="store_false")
    args = parser.parse_args()

    import ultralytics
    from ultralytics import YOLO

    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "device": args.device,
        "dynamic": False,
        "simplify": True,
        "nms": args.nms,
    }
    if args.format == "onnx":
        export_args["opset"] = args.opset
    if args.precision == "fp16":
        export_args["quantize"] = 16
    elif args.precision == "int8":
        export_args.update(
            quantize=8,
            data=str(resolve_path(args.data)),
            fraction=args.fraction,
        )

    exported = Path(YOLO(str(model_path)).export(**export_args))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(model_path),
        "exported_model": str(exported.resolve()),
        "format": args.format,
        "precision": args.precision,
        "imgsz": args.imgsz,
        "nms": args.nms,
        "calibration_data": str(resolve_path(args.data)) if args.precision == "int8" else None,
        "calibration_fraction": args.fraction if args.precision == "int8" else None,
        "model_size_bytes": exported.stat().st_size,
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "arguments": export_args,
    }
    metadata_path = exported.with_suffix(exported.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
