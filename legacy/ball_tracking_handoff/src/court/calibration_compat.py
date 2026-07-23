"""
Load court keypoints from a static calibration YAML file.

Format: standard calibration YAML, mapping keypoint names to [x, y].
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from .observation import CourtObservation

# Name → index mapping (matching layout.py's 14-point schema)
NAME_TO_IDX = {
    "far_baseline_left": 0,
    "far_baseline_center": 1,
    "far_baseline_right": 2,
    "far_kitchen_left": 3,
    "far_kitchen_center": 4,
    "far_kitchen_right": 5,
    "net_left": 6,
    "net_right": 7,
    "near_kitchen_left": 8,
    "near_kitchen_center": 9,
    "near_kitchen_right": 10,
    "near_baseline_left": 11,
    "near_baseline_center": 12,
    "near_baseline_right": 13,
}


def load_court_observation(yaml_path: str) -> Optional[CourtObservation]:
    """Load court keypoints from a calibration YAML file.

    The YAML should have a ``court.keypoints_14`` dict mapping
    keypoint names (e.g. "far_baseline_left") to ``[x, y]`` pixel coords,
    or a top-level dict with the same keys.

    Returns ``CourtObservation`` or None if the file is missing/invalid.
    """
    path = Path(yaml_path)
    if not path.exists():
        print(f"[court] calibration file not found: {yaml_path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    # Try court.keypoints_14 first
    kps_raw = data.get("court", {}).get("keypoints_14", None)
    if kps_raw is None:
        # Try top-level keys directly
        kps_raw = {k: v for k, v in data.items() if k in NAME_TO_IDX}

    if not kps_raw:
        return None

    keypoints: dict[int, tuple[float, float]] = {}
    for name, idx in NAME_TO_IDX.items():
        if name in kps_raw:
            pt = kps_raw[name]
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
                if x > 0 or y > 0:  # skip placeholder (0,0)
                    keypoints[idx] = (x, y)

    if not keypoints:
        return None

    return CourtObservation.from_calibration(keypoints)
