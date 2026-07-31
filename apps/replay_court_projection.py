"""CLI for re-rendering court projection from one completed R9 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.dual_camera.projection_replay import (
    load_projection_replay_source,
    replay_court_projection,
)
from src.tracking.factory import load_config, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render court projection from saved R9 video/JSONL without "
            "running detector or tracker inference"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/tracking.yaml",
    )
    parser.add_argument(
        "--source-run",
        required=True,
        help="Completed dual-camera run directory",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="New replay run ID; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/court_projection_replay",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help=(
            "Write only the blank court panel; do not decode the source MP4"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    source = load_projection_replay_source(project_path(args.source_run))
    replay_court_projection(
        source,
        output_dir=project_path(args.output_dir),
        run_id=args.run_id,
        config=config,
        config_path=config_path,
        max_frames=args.max_frames,
        projection_only=args.projection_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
