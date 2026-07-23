"""
Court keypoint observation for a single frame.

Holds detected or manually-annotated keypoints in image pixel coordinates.
Can be empty — the mini-court always draws regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CourtObservation:
    """Per-frame court keypoint data.

    All fields are optional / may be empty — the system must work
    even when no keypoints are available.
    """

    # Keypoints in image pixels: {kp_idx: (x, y)}
    keypoints: dict[int, tuple[float, float]] = field(default_factory=dict)

    # Per-keypoint confidence: {kp_idx: 0.0–1.0}
    confidences: dict[int, float] = field(default_factory=dict)

    # How these keypoints were obtained
    source: str = "none"  # "detector" | "manual" | "calibration_file" | "none"

    @property
    def num_visible(self) -> int:
        """Number of valid keypoints available."""
        return len(self.keypoints)

    @property
    def is_reliable(self) -> bool:
        """At least 4 keypoints — minimum for homography."""
        return self.num_visible >= 4

    @property
    def visible_indices(self) -> list[int]:
        """Sorted list of keypoint indices that are present."""
        return sorted(self.keypoints.keys())

    # -- factory methods --------------------------------------------------

    @classmethod
    def empty(cls) -> CourtObservation:
        """Create an empty observation (no keypoints available)."""
        return cls(keypoints={}, confidences={}, source="none")

    @classmethod
    def from_manual(
        cls,
        kp_dict: dict[int, tuple[float, float]],
        confidences: Optional[dict[int, float]] = None,
    ) -> CourtObservation:
        """Create from manually-annotated keypoints."""
        confs = confidences or {k: 1.0 for k in kp_dict}
        return cls(keypoints=dict(kp_dict), confidences=confs, source="manual")

    @classmethod
    def from_calibration(
        cls,
        kp_dict: dict[int, tuple[float, float]],
    ) -> CourtObservation:
        """Create from a pre-calibrated keypoint file."""
        confs = {k: 1.0 for k in kp_dict}
        return cls(keypoints=dict(kp_dict), confidences=confs, source="calibration_file")

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSONL."""
        return {
            "source": self.source,
            "num_visible": self.num_visible,
            "is_reliable": self.is_reliable,
            "visible_indices": self.visible_indices,
            "keypoints": {
                str(k): list(v) for k, v in self.keypoints.items()
            },
        }
