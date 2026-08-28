"""CLI for synchronized left/right half-court tracking."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.dual_camera.artifacts import DualRunArtifacts
from src.tracking.dual_camera.runner import run_pair
from src.tracking.factory import (
    build_detector,
    build_person_detector,
    load_config,
    project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track synchronized left/right half-court video pairs"
    )
    parser.add_argument(
        "--config",
        default="configs/tracking.yaml",
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("RUN_ID", "LEFT_VIDEO", "RIGHT_VIDEO"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/dual_camera",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--max-players-per-half",
        type=int,
        default=2,
        help="Maximum eligible players selected independently in each half",
    )
    output_policy = parser.add_mutually_exclusive_group()
    output_policy.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a pair when all final artifacts already exist",
    )
    output_policy.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace all final artifacts for the same run ID",
    )
    parser.add_argument(
        "--global-switch-min-missing-ms",
        type=float,
        default=20.0,
        help="Minimum active-side missing time before switching cameras",
    )
    parser.add_argument(
        "--run-note",
        help="Short experiment purpose recorded in the run manifest",
    )
    parser.add_argument(
        "--no-court-projection",
        action="store_true",
        help="Disable the read-only court projection and its output panel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_players_per_half < 1:
        raise ValueError("--max-players-per-half must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    config_path = project_path(args.config)
    config = copy.deepcopy(load_config(config_path))
    if args.no_court_projection:
        config.setdefault("runtime", {}).setdefault(
            "court_projection",
            {},
        )["enabled"] = False
    person_config = config.setdefault("runtime", {}).setdefault(
        "person_detection",
        {},
    )
    person_config.setdefault("player_selection", {})["max_players"] = (
        args.max_players_per_half
    )
    config.setdefault("tracker", {})["max_output_tracks"] = 0
    config["run_context"] = {
        "entrypoint": "apps/track_dual_halves.py",
        "effective_overrides": {
            "runtime.court_projection.enabled": bool(
                config.get("runtime", {})
                .get("court_projection", {})
                .get("enabled", False)
            ),
            "runtime.person_detection.player_selection.max_players": (
                args.max_players_per_half
            ),
            "tracker.max_output_tracks": 0,
        },
    }

    print(f"[players] max eligible players per half={args.max_players_per_half}")
    runtime_config = config.get("runtime", {})
    referee_output_enabled = bool(
        runtime_config.get("court_projection", {}).get("enabled", False)
        and runtime_config.get("referee", {}).get("enabled", False)
    )
    detector = build_detector(config)
    person_detector = build_person_detector(config)
    output_dir = project_path(args.output_dir)

    for run_id, left_value, right_value in args.pair:
        artifacts = DualRunArtifacts.for_run(output_dir, run_id)
        existing = [
            path for path in artifacts.all_files() if path.exists()
        ]
        required = artifacts.required_files(
            referee_enabled=referee_output_enabled,
        )
        stable_complete = all(
            path.exists() for path in required
        )
        if args.skip_existing and stable_complete:
            print(f"[skip:{run_id}] all final artifacts already exist")
            continue
        if existing and not args.overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Run '{run_id}' already has final artifacts: {names}. "
                "Use a new run ID, --skip-existing, or explicit --overwrite."
            )
        if args.overwrite:
            artifacts.remove()

        run_pair(
            run_id,
            project_path(left_value),
            project_path(right_value),
            output_dir,
            config,
            detector,
            person_detector,
            config_path=config_path,
            max_frames=args.max_frames,
            global_switch_min_missing_ms=args.global_switch_min_missing_ms,
            run_note=args.run_note,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
