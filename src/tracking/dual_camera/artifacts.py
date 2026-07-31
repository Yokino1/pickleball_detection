"""Output naming, partial-file handling and validation for dual-camera runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class DualRunArtifacts:
    video: Path
    left_jsonl: Path
    right_jsonl: Path
    global_jsonl: Path
    manifest: Path

    @classmethod
    def for_run(cls, output_dir: Path, run_id: str) -> "DualRunArtifacts":
        if (
            not run_id
            or run_id in {".", ".."}
            or "/" in run_id
            or "\\" in run_id
        ):
            raise ValueError(
                "run_id must be one non-empty directory name without separators"
            )
        run_dir = output_dir / run_id
        return cls(
            video=run_dir / "dual_tracking.mp4",
            left_jsonl=run_dir / "left_tracking.jsonl",
            right_jsonl=run_dir / "right_tracking.jsonl",
            global_jsonl=run_dir / "global_tracking.jsonl",
            manifest=run_dir / "manifest.json",
        )

    @property
    def run_dir(self) -> Path:
        return self.video.parent

    def final_files(self) -> tuple[Path, ...]:
        return (
            self.video,
            self.left_jsonl,
            self.right_jsonl,
            self.global_jsonl,
            self.manifest,
        )

    def partial(self) -> "DualRunArtifacts":
        return DualRunArtifacts(
            video=self.video.with_name(
                self.video.name.replace(".mp4", ".partial.mp4")
            ),
            left_jsonl=self.left_jsonl.with_name(
                self.left_jsonl.name.replace(".jsonl", ".partial.jsonl")
            ),
            right_jsonl=self.right_jsonl.with_name(
                self.right_jsonl.name.replace(".jsonl", ".partial.jsonl")
            ),
            global_jsonl=self.global_jsonl.with_name(
                self.global_jsonl.name.replace(".jsonl", ".partial.jsonl")
            ),
            manifest=self.manifest.with_name(
                self.manifest.name.replace(".json", ".partial.json")
            ),
        )

    def remove(self) -> None:
        for path in self.final_files():
            path.unlink(missing_ok=True)

    def promote_from(self, partial: "DualRunArtifacts") -> None:
        for source, target in zip(
            partial.final_files(),
            self.final_files(),
            strict=True,
        ):
            source.replace(target)


def validate_written_outputs(
    artifacts: DualRunArtifacts,
    *,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int],
) -> None:
    capture = cv2.VideoCapture(str(artifacts.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen written video: {artifacts.video}")
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"Written video frame mismatch: expected={expected_frames}, "
            f"actual={actual_frames}"
        )
    if abs(actual_fps - expected_fps) > 1e-3:
        raise RuntimeError(
            f"Written video FPS mismatch: expected={expected_fps}, "
            f"actual={actual_fps}"
        )
    if actual_size != expected_size:
        raise RuntimeError(
            f"Written video size mismatch: expected={expected_size}, "
            f"actual={actual_size}"
        )
    for stream, path in (
        ("left", artifacts.left_jsonl),
        ("right", artifacts.right_jsonl),
        ("global", artifacts.global_jsonl),
    ):
        with path.open("r", encoding="utf-8") as handle:
            line_count = sum(1 for _ in handle)
        if line_count != expected_frames:
            raise RuntimeError(
                f"{stream} JSONL frame mismatch: expected={expected_frames}, "
                f"actual={line_count}"
            )
