"""Output naming, partial-file handling and validation for dual-camera runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class DualRunArtifacts:
    video: Path
    left_jsonl: Path
    right_jsonl: Path
    global_jsonl: Path
    rally_results_jsonl: Path
    manifest: Path

    @classmethod
    def for_run(cls, output_dir: Path, run_id: str) -> DualRunArtifacts:
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
            rally_results_jsonl=run_dir / "rally_results.jsonl",
            manifest=run_dir / "manifest.json",
        )

    @property
    def run_dir(self) -> Path:
        return self.video.parent

    def final_files(self) -> tuple[Path, ...]:
        """Return the stable R9 artifact contract.

        ``rally_results.jsonl`` is a downstream demo output and intentionally
        stays outside this tuple so completed pre-referee runs remain valid.
        """
        return (
            self.video,
            self.left_jsonl,
            self.right_jsonl,
            self.global_jsonl,
            self.manifest,
        )

    def all_files(self) -> tuple[Path, ...]:
        return (*self.final_files(), self.rally_results_jsonl)

    def required_files(self, *, referee_enabled: bool) -> tuple[Path, ...]:
        """Return the capability-aware completion contract for this run."""
        return self.all_files() if referee_enabled else self.final_files()

    def partial(self) -> DualRunArtifacts:
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
            rally_results_jsonl=self.rally_results_jsonl.with_name(
                self.rally_results_jsonl.name.replace(
                    ".jsonl",
                    ".partial.jsonl",
                )
            ),
            manifest=self.manifest.with_name(
                self.manifest.name.replace(".json", ".partial.json")
            ),
        )

    def remove(self) -> None:
        for path in self.all_files():
            path.unlink(missing_ok=True)

    def promote_from(self, partial: DualRunArtifacts) -> None:
        # Publish the manifest last: a visible completed manifest must never
        # point at an event stream that has not been promoted yet.
        pairs = [
            (partial.video, self.video),
            (partial.left_jsonl, self.left_jsonl),
            (partial.right_jsonl, self.right_jsonl),
            (partial.global_jsonl, self.global_jsonl),
        ]
        if partial.rally_results_jsonl.exists():
            pairs.append(
                (partial.rally_results_jsonl, self.rally_results_jsonl)
            )
        pairs.append((partial.manifest, self.manifest))
        existing = [target for _source, target in pairs if target.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing run artifacts: "
                + ", ".join(str(path) for path in existing)
            )

        promoted: list[Path] = []
        try:
            for source, target in pairs:
                source.replace(target)
                promoted.append(target)
        except BaseException:
            for target in promoted:
                target.unlink(missing_ok=True)
            raise


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
    validate_rally_result_contract(
        artifacts.global_jsonl,
        artifacts.rally_results_jsonl,
    )


def validate_rally_result_contract(
    global_jsonl: Path,
    rally_results_jsonl: Path,
) -> dict:
    """Validate the optional downstream rally stream against frame snapshots.

    Legacy runs may have neither a top-level referee snapshot nor a separate
    rally stream. Once referee snapshots are present, the rally stream becomes
    required and must be an exact one-row copy of every one-shot terminal event.
    """
    terminal_results: list[dict] = []
    referee_frames = 0
    final_score: dict[str, int] | None = None
    final_service_owner_side: str | None = None
    with global_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _jsonl_object(line, global_jsonl, line_number)
            referee = record.get("referee")
            if referee is None:
                continue
            if not isinstance(referee, dict):
                raise RuntimeError(
                    f"Invalid referee snapshot at {global_jsonl}:{line_number}"
                )
            referee_frames += 1
            final_score = _score_mapping(
                referee.get("score"),
                context=f"{global_jsonl}:{line_number} referee.score",
            )
            if "service_owner_side" in referee:
                owner = referee.get("service_owner_side")
                final_service_owner_side = (
                    str(owner) if owner is not None else None
                )
            rally_result = referee.get("rally_result")
            if rally_result is not None:
                if not isinstance(rally_result, dict):
                    raise RuntimeError(
                        "Invalid referee.rally_result at "
                        f"{global_jsonl}:{line_number}"
                    )
                terminal_results.append(rally_result)

    if not rally_results_jsonl.is_file():
        if referee_frames:
            raise RuntimeError(
                "Referee snapshots require rally_results.jsonl: "
                f"{rally_results_jsonl}"
            )
        return {
            "enabled": False,
            "referee_frames": 0,
            "rally_results": 0,
            "final_score": None,
            "final_service_owner_side": None,
        }

    rally_results: list[dict] = []
    with rally_results_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            rally_results.append(
                _jsonl_object(line, rally_results_jsonl, line_number)
            )

    if not referee_frames and rally_results:
        raise RuntimeError(
            "rally_results.jsonl contains rows without referee snapshots"
        )
    if terminal_results != rally_results:
        raise RuntimeError(
            "rally_results.jsonl does not match one-shot referee terminal "
            f"snapshots: frames={len(terminal_results)}, "
            f"rows={len(rally_results)}"
        )

    seen_rally_ids: set[int] = set()
    seen_result_ids: set[str] = set()
    previous_score: dict[str, int] | None = None
    previous_next_server_side: str | None = None
    service_ownership_seen = False
    for index, result in enumerate(rally_results, start=1):
        try:
            rally_id = int(result["rally_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid rally_id in {rally_results_jsonl} row {index}"
            ) from error
        if rally_id in seen_rally_ids:
            raise RuntimeError(f"Duplicate rally_id in rally results: {rally_id}")
        seen_rally_ids.add(rally_id)
        result_id = result.get("result_id")
        if result_id is not None:
            result_id = str(result_id)
            expected_result_id = f"rally-{rally_id:06d}"
            if result_id != expected_result_id:
                raise RuntimeError(
                    f"Invalid result_id in rally results: {result_id!r}; "
                    f"expected={expected_result_id!r}"
                )
            if result_id in seen_result_ids:
                raise RuntimeError(
                    f"Duplicate result_id in rally results: {result_id}"
                )
            seen_result_ids.add(result_id)

        before = _score_mapping(
            result.get("score_before"),
            context=f"{rally_results_jsonl}:{index} score_before",
        )
        after = _score_mapping(
            result.get("score_after"),
            context=f"{rally_results_jsonl}:{index} score_after",
        )
        if previous_score is not None and before != previous_score:
            raise RuntimeError(
                f"Non-contiguous score history at rally result row {index}"
            )
        awarded_to = result.get("point_awarded_to")
        expected_after = dict(before)
        if awarded_to is not None:
            awarded_to = str(awarded_to)
            if awarded_to not in expected_after:
                raise RuntimeError(
                    f"Unknown point_awarded_to={awarded_to!r} at row {index}"
                )
            expected_after[awarded_to] += 1
        if after != expected_after:
            raise RuntimeError(
                f"Invalid score transition at rally result row {index}"
            )
        if result.get("next_server_side") is not None:
            service_ownership_seen = True
            server_value = result.get("server_side")
            server_side = (
                str(server_value) if server_value is not None else None
            )
            next_value = result.get("next_server_side")
            next_server_side = (
                str(next_value) if next_value is not None else None
            )
            if (
                previous_next_server_side is not None
                and server_side != previous_next_server_side
            ):
                raise RuntimeError(
                    "Server side does not match prior service owner at "
                    f"rally result row {index}"
                )
            winner_value = result.get("rally_winner")
            winner_side = (
                str(winner_value) if winner_value is not None else None
            )
            status = str(result.get("status", ""))
            if status in {"confirmed", "demo_inferred"}:
                if winner_side is not None and next_server_side != winner_side:
                    raise RuntimeError(
                        "Resolvable rally winner must own the next serve at "
                        f"rally result row {index}"
                    )
            elif status == "unresolved" and next_server_side != server_side:
                raise RuntimeError(
                    "Unresolved rally must retain the current server at "
                    f"rally result row {index}"
                )
            if (
                next_server_side is not None
                and next_server_side not in after
            ):
                raise RuntimeError(
                    f"Unknown next_server_side={next_server_side!r} "
                    f"at row {index}"
                )
            previous_next_server_side = next_server_side
        previous_score = after

    if rally_results and final_score != previous_score:
        raise RuntimeError(
            "Final referee score does not match the last rally score_after"
        )
    if (
        service_ownership_seen
        and final_service_owner_side != previous_next_server_side
    ):
        raise RuntimeError(
            "Final referee service owner does not match the last rally "
            "next_server_side"
        )
    return {
        "enabled": referee_frames > 0,
        "referee_frames": referee_frames,
        "rally_results": len(rally_results),
        "final_score": final_score,
        "final_service_owner_side": final_service_owner_side,
    }


def _jsonl_object(line: str, path: Path, line_number: int) -> dict:
    if not line.strip():
        raise RuntimeError(f"Blank JSONL row at {path}:{line_number}")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON at {path}:{line_number}: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSONL row must be an object at {path}:{line_number}")
    return value


def _score_mapping(value, *, context: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"Invalid score mapping at {context}")
    score: dict[str, int] = {}
    for team, points in value.items():
        if isinstance(points, bool) or not isinstance(points, int) or points < 0:
            raise RuntimeError(f"Invalid score value at {context}: {points!r}")
        score[str(team)] = points
    return score
