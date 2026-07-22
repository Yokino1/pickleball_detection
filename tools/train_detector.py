"""Train a reproducible one-class pickleball detector with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--data", default="datasets/cleaned_ball_detection/data.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="pickleball_nano_960")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = args.model
    local_model = Path(model_path)
    if local_model.exists() or (PROJECT_ROOT / local_model).exists():
        model_path = resolve_path(model_path)
    model = YOLO(model_path)
    model.train(
        data=resolve_path(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "artifacts" / "training"),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        patience=30,
        close_mosaic=10,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
