"""Re-render court projection from a completed R9 run without model inference."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2

from ...court import build_court_projection
from ...referee import REFEREE_CONTRACT_VERSION, build_referee_engine
from ..run_manifest import (
    complete_manifest,
    create_manifest,
    sha256_file,
    write_manifest,
)
from ..types import BallTrack
from .artifacts import validate_rally_result_contract
from .coordinator import GlobalBallSelection
from .rendering import (
    HEADER_HEIGHT,
    make_header,
)
from .rendering import (
    court_motion_discontinuity_reason as _court_motion_discontinuity_reason,
)


@dataclass(frozen=True)
class ProjectionReplaySource:
    """Validated files and geometry from one completed dual-camera run."""

    run_dir: Path
    run_id: str
    rendered_video: Path
    left_jsonl: Path
    right_jsonl: Path
    global_jsonl: Path
    left_size: tuple[int, int]
    right_size: tuple[int, int]
    fps: float
    header_height: int
    frame_scale_overrides: dict[str, float]
    manifest: dict
    expected_frames: int | None = None
    source_manifest_sha256: str | None = None
    artifact_identities: dict[str, dict] = field(default_factory=dict)

    @property
    def tracking_width(self) -> int:
        return self.left_size[0] + self.right_size[0]

    @property
    def frame_height(self) -> int:
        return self.left_size[1]


@dataclass(frozen=True)
class ProjectionReplayArtifacts:
    """Independent outputs derived from one immutable source run."""

    run_dir: Path
    video: Path
    global_jsonl: Path
    rally_results_jsonl: Path
    manifest: Path

    @classmethod
    def for_run(
        cls,
        output_dir: Path,
        run_id: str,
    ) -> ProjectionReplayArtifacts:
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
            run_dir=run_dir,
            video=run_dir / "projection_replay.mp4",
            global_jsonl=run_dir / "global_projection.jsonl",
            rally_results_jsonl=run_dir / "rally_results.jsonl",
            manifest=run_dir / "manifest.json",
        )

    def partial_video(self) -> Path:
        return self.video.with_name("projection_replay.partial.mp4")

    def partial_jsonl(self) -> Path:
        return self.global_jsonl.with_name("global_projection.partial.jsonl")

    def partial_manifest(self) -> Path:
        return self.manifest.with_name("manifest.partial.json")

    def partial_rally_results_jsonl(self) -> Path:
        return self.rally_results_jsonl.with_name(
            "rally_results.partial.jsonl"
        )

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.video,
            self.global_jsonl,
            self.rally_results_jsonl,
            self.manifest,
            self.partial_video(),
            self.partial_jsonl(),
            self.partial_rally_results_jsonl(),
            self.partial_manifest(),
        )


def load_projection_replay_source(
    run_dir: Path,
    *,
    require_rendered_video: bool = True,
) -> ProjectionReplaySource:
    """Load one completed source run and resolve required replay artifacts."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "completed":
        raise ValueError(f"Source run is not completed: {manifest_path}")

    input_by_side = {
        item.get("side"): item
        for item in manifest.get("inputs", [])
        if item.get("side") in {"left", "right"}
    }
    if set(input_by_side) != {"left", "right"}:
        raise ValueError("Source manifest must describe left and right inputs")
    left_size = _manifest_frame_size(input_by_side["left"], "left")
    right_size = _manifest_frame_size(input_by_side["right"], "right")
    if left_size[1] != right_size[1]:
        raise ValueError("Source left/right frame heights must match")

    outputs = manifest.get("outputs", {})
    rendered_video = _resolve_output_path(
        outputs,
        "video",
        run_dir / "dual_tracking.mp4",
    )
    left_jsonl = _resolve_output_path(
        outputs,
        "left_jsonl",
        run_dir / "left_tracking.jsonl",
    )
    right_jsonl = _resolve_output_path(
        outputs,
        "right_jsonl",
        run_dir / "right_tracking.jsonl",
    )
    global_jsonl = _resolve_output_path(
        outputs,
        "global_jsonl",
        run_dir / "global_tracking.jsonl",
    )
    required_paths = [left_jsonl, right_jsonl, global_jsonl]
    if require_rendered_video:
        required_paths.insert(0, rendered_video)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Source replay artifact missing: {path}")

    declared_frames = _declared_source_frames(manifest, input_by_side)
    validated_frames = _validate_source_jsonl_streams(
        left_jsonl,
        right_jsonl,
        global_jsonl,
        expected_frames=declared_frames,
    )
    expected_frames = (
        declared_frames
        if declared_frames is not None
        else (validated_frames if validated_frames > 0 else None)
    )
    artifact_identities = {
        "left_jsonl": _artifact_identity(
            left_jsonl,
            outputs.get("left_jsonl"),
            role="left_jsonl",
        ),
        "right_jsonl": _artifact_identity(
            right_jsonl,
            outputs.get("right_jsonl"),
            role="right_jsonl",
        ),
        "global_jsonl": _artifact_identity(
            global_jsonl,
            outputs.get("global_jsonl"),
            role="global_jsonl",
        ),
    }
    if require_rendered_video:
        artifact_identities["video"] = _artifact_identity(
            rendered_video,
            outputs.get("video"),
            role="video",
        )

    parameters = manifest.get("parameters", {})
    scales = dict(parameters.get("frame_scale_overrides", {}))
    fallback_scale = (left_size[0] + right_size[0]) / 1280.0
    frame_scale_overrides = {
        side: max(1e-6, float(scales.get(side, fallback_scale)))
        for side in ("left", "right")
    }
    fps = float(
        manifest.get("summary", {}).get(
            "source_fps",
            input_by_side["left"].get("fps", 0.0),
        )
    )
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Source FPS must be positive")
    return ProjectionReplaySource(
        run_dir=run_dir,
        run_id=str(manifest.get("run_id") or run_dir.name),
        rendered_video=rendered_video,
        left_jsonl=left_jsonl,
        right_jsonl=right_jsonl,
        global_jsonl=global_jsonl,
        left_size=left_size,
        right_size=right_size,
        fps=fps,
        header_height=int(parameters.get("header_height", HEADER_HEIGHT)),
        frame_scale_overrides=frame_scale_overrides,
        manifest=manifest,
        expected_frames=expected_frames,
        source_manifest_sha256=sha256_file(manifest_path),
        artifact_identities=artifact_identities,
    )


def replay_court_projection(
    source: ProjectionReplaySource,
    *,
    output_dir: Path,
    run_id: str,
    config: dict,
    config_path: Path,
    max_frames: int | None = None,
    projection_only: bool = False,
) -> ProjectionReplayArtifacts:
    """Replace the source run's court panel using saved R9 records only."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    artifacts = ProjectionReplayArtifacts.for_run(output_dir, run_id)
    existing = [path for path in artifacts.all_paths() if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Replay run '{run_id}' already has artifacts: {names}"
        )

    projector, renderer, event_interpreter = build_court_projection(
        config,
        {
            "left": source.left_size,
            "right": source.right_size,
        },
    )
    if (
        projector is None
        or renderer is None
        or event_interpreter is None
    ):
        raise ValueError("Court projection must be enabled for replay")
    referee_engine = build_referee_engine(config, projector.layout)

    panel_width = renderer.panel_width(source.frame_height)
    expected_height = source.frame_height + source.header_height
    capture = None
    if not projection_only:
        capture = cv2.VideoCapture(str(source.rendered_video))
        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open source rendered video: {source.rendered_video}"
            )
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width < source.tracking_width:
            capture.release()
            raise ValueError(
                "Source rendered video is narrower than the two tracking frames"
            )
        if source_height != expected_height:
            capture.release()
            raise ValueError(
                "Source rendered height mismatch: "
                f"expected={expected_height}, actual={source_height}"
            )
    output_size = (
        (panel_width, source.frame_height)
        if projection_only
        else (
            source.tracking_width + panel_width,
            expected_height,
        )
    )
    artifacts.run_dir.mkdir(parents=True, exist_ok=False)
    codec = str(config.get("output", {}).get("codec", "mp4v"))
    writer = cv2.VideoWriter(
        str(artifacts.partial_video()),
        cv2.VideoWriter_fourcc(*codec),
        source.fps,
        output_size,
    )
    if not writer.isOpened():
        if capture is not None:
            capture.release()
        writer.release()
        _remove_failed_replay(artifacts)
        raise RuntimeError(
            f"Cannot open replay video writer: {artifacts.partial_video()}"
        )

    try:
        manifest_inputs = [
            {
                "role": "source_run",
                "run_id": source.run_id,
                "path": str(source.run_dir),
                "manifest_sha256": source.source_manifest_sha256,
            }
        ]
        if not projection_only:
            manifest_inputs.append(
                {
                    "role": "source_rendered_video",
                    "path": str(source.rendered_video),
                    **source.artifact_identities.get("video", {}),
                }
            )
        manifest_inputs.extend(
            [
                {
                    "role": "source_left_jsonl",
                    "path": str(source.left_jsonl),
                    **source.artifact_identities.get("left_jsonl", {}),
                },
                {
                    "role": "source_right_jsonl",
                    "path": str(source.right_jsonl),
                    **source.artifact_identities.get("right_jsonl", {}),
                },
                {
                    "role": "source_global_jsonl",
                    "path": str(source.global_jsonl),
                    **source.artifact_identities.get("global_jsonl", {}),
                },
            ]
        )
        manifest = create_manifest(
            project_root=config_path.resolve().parents[1],
            run_id=run_id,
            run_type="court_projection_replay",
            config_path=config_path,
            config=config,
            inputs=manifest_inputs,
            parameters={
                "model_inference": False,
                "projection_only": projection_only,
                "source_run_id": source.run_id,
                "source_expected_frames": source.expected_frames,
                "max_frames": max_frames,
                "source_fps": source.fps,
                "tracking_width": source.tracking_width,
                "court_panel_width": panel_width,
                "referee_enabled": referee_engine is not None,
                "referee_contract_version": (
                    REFEREE_CONTRACT_VERSION
                    if referee_engine is not None
                    else None
                ),
                "referee_scoring_mode": (
                    referee_engine.score_recorder.scoring_mode
                    if referee_engine is not None
                    else None
                ),
                "referee_service_ownership_enabled": (
                    referee_engine.service_ownership_enabled
                    if referee_engine is not None
                    else None
                ),
                "header_height": source.header_height,
                "frame_scale_overrides": source.frame_scale_overrides,
            },
        )
        write_manifest(artifacts.partial_manifest(), manifest)
    except BaseException:
        if capture is not None:
            capture.release()
        writer.release()
        _remove_failed_replay(artifacts)
        raise

    processed = 0
    bounce_frames = 0
    hit_frames = 0
    outside_bounce_frames = 0
    second_bounce_frames = 0
    started = time.perf_counter()
    try:
        with (
            source.left_jsonl.open("r", encoding="utf-8") as left_handle,
            source.right_jsonl.open("r", encoding="utf-8") as right_handle,
            source.global_jsonl.open("r", encoding="utf-8") as global_handle,
            artifacts.partial_jsonl().open(
                "w",
                encoding="utf-8",
            ) as output_handle,
            artifacts.partial_rally_results_jsonl().open(
                "w",
                encoding="utf-8",
            ) as rally_results_handle,
        ):
            while max_frames is None or processed < max_frames:
                if capture is not None:
                    ok, frame = capture.read()
                else:
                    ok, frame = True, None
                left_line = left_handle.readline()
                right_line = right_handle.readline()
                global_line = global_handle.readline()
                ended = [
                    not left_line,
                    not right_line,
                    not global_line,
                ]
                if capture is not None:
                    ended.insert(0, not ok)
                if all(ended):
                    break
                if any(ended):
                    raise RuntimeError(
                        "Source video and JSONL frame counts do not match"
                    )

                left_record = json.loads(left_line)
                right_record = json.loads(right_line)
                global_record = json.loads(global_line)
                frame_index = int(global_record["frame_index"])
                if (
                    int(left_record["frame_index"]) != frame_index
                    or int(right_record["frame_index"]) != frame_index
                ):
                    raise RuntimeError(
                        f"Source frame index mismatch at replay row {processed}"
                    )

                selection = _selection_from_record(
                    global_record.get("global_ball", {})
                )
                projection = projector.project_track(
                    selection.active_side,
                    selection.track,
                )
                reset_reason = (
                    global_record.get("rendering", {})
                    .get("trail_reset_reason")
                )
                court_discontinuity_reason = (
                    _court_motion_discontinuity_reason(reset_reason)
                )
                if (
                    reset_reason is not None
                    and reset_reason != "camera_side_switch"
                ):
                    renderer.reset()
                active_record = (
                    left_record
                    if selection.active_side == "left"
                    else right_record
                )
                eligible_players = active_record.get("players", [])
                eligible_player_centers_court_xy = (
                    projector.project_eligible_player_box_centers(
                        selection.active_side,
                        eligible_players,
                    )
                )
                projection.event = event_interpreter.update(
                    projection,
                    track=selection.track,
                    active_side=selection.active_side,
                    local_track_id=selection.local_track_id,
                    tracker_diagnostics={
                        "left": left_record.get("diagnostics", {}).get(
                            "tracker",
                            {},
                        ),
                        "right": right_record.get("diagnostics", {}).get(
                            "tracker",
                            {},
                        ),
                    },
                    frame_scale_overrides=source.frame_scale_overrides,
                    discontinuity_reason=court_discontinuity_reason,
                    timestamp_s=_record_timestamp_s(
                        global_record,
                        frame_index,
                        source.fps,
                    ),
                    frame_index=frame_index,
                    eligible_players=eligible_players,
                ).to_dict()
                referee_result = (
                    referee_engine.update(
                        projection,
                        timestamp_s=_record_timestamp_s(
                            global_record,
                            frame_index,
                            source.fps,
                        ),
                        frame_index=frame_index,
                        discontinuity_reason=court_discontinuity_reason,
                        eligible_player_centers_court_xy=(
                            eligible_player_centers_court_xy
                        ),
                    )
                    if referee_engine is not None
                    else None
                )
                referee_record = (
                    referee_result.to_dict()
                    if referee_result is not None
                    else None
                )
                rally_result = (
                    referee_record.get("rally_result")
                    if referee_record is not None
                    else None
                )
                if rally_result is not None:
                    rally_results_handle.write(
                        json.dumps(rally_result, ensure_ascii=False) + "\n"
                    )

                event_names = set(
                    (projection.event or {}).get("events", [])
                )
                bounce_frames += int("bounce_candidate" in event_names)
                hit_frames += int("paddle_hit_candidate" in event_names)
                outside_bounce_frames += int(
                    "out_of_bounds_bounce_candidate" in event_names
                )
                second_bounce_frames += int(
                    "second_bounce_candidate" in event_names
                )

                court_panel = renderer.render(
                    source.frame_height,
                    projection,
                    referee=referee_record,
                )
                if projection_only:
                    combined = court_panel
                else:
                    assert frame is not None
                    tracking_body = frame[
                        source.header_height:,
                        :source.tracking_width,
                    ]
                    combined_body = cv2.hconcat(
                        [tracking_body, court_panel]
                    )
                    header = make_header(
                        combined_body.shape[1],
                        source.left_size[0],
                        selection,
                        court_panel_width=panel_width,
                    )
                    combined = cv2.vconcat([header, combined_body])
                    cv2.line(
                        combined,
                        (source.tracking_width, 0),
                        (source.tracking_width, combined.shape[0] - 1),
                        (255, 160, 0),
                        2,
                    )
                writer.write(combined)

                derived_record = dict(global_record)
                derived_record["court"] = projection.to_dict()
                derived_record["referee"] = referee_record
                derived_record["projection_replay"] = {
                    "source_run_id": source.run_id,
                    "source_frame_index": frame_index,
                    "model_inference": False,
                }
                output_handle.write(
                    json.dumps(derived_record, ensure_ascii=False) + "\n"
                )
                processed += 1
                if processed % 500 == 0:
                    print(f"[replay:{run_id}] frames={processed}")
    except BaseException:
        if capture is not None:
            capture.release()
        writer.release()
        _remove_failed_replay(artifacts)
        raise
    finally:
        if capture is not None:
            capture.release()
        writer.release()

    expected_processed = (
        None
        if source.expected_frames is None
        else (
            source.expected_frames
            if max_frames is None
            else min(source.expected_frames, max_frames)
        )
    )
    if expected_processed is not None and processed != expected_processed:
        _remove_failed_replay(artifacts)
        raise RuntimeError(
            "Projection replay source frame mismatch: "
            f"expected={expected_processed}, processed={processed}"
        )
    if processed < 1:
        _remove_failed_replay(artifacts)
        raise RuntimeError("Projection replay produced no frames")
    elapsed = time.perf_counter() - started
    try:
        rally_contract = _validate_replay_outputs(
            artifacts.partial_video(),
            artifacts.partial_jsonl(),
            artifacts.partial_rally_results_jsonl(),
            expected_frames=processed,
            expected_fps=source.fps,
            expected_size=output_size,
        )
        if referee_engine is not None:
            referee_diagnostics = referee_engine.diagnostics()
            expected_score = referee_diagnostics.get("score")
            if rally_contract.get("final_score") != expected_score:
                raise RuntimeError(
                    "Replay referee diagnostics score does not match JSONL: "
                    f"expected={expected_score}, "
                    f"actual={rally_contract.get('final_score')}"
                )
            expected_service_owner = referee_diagnostics.get(
                "service_owner_side"
            )
            if (
                rally_contract.get("final_service_owner_side")
                != expected_service_owner
            ):
                raise RuntimeError(
                    "Replay referee diagnostics service owner does not match "
                    "JSONL: "
                    f"expected={expected_service_owner}, "
                    "actual="
                    f"{rally_contract.get('final_service_owner_side')}"
                )
        completed = complete_manifest(
            manifest,
            outputs={
                "video": {
                    "path": str(artifacts.video.resolve()),
                    "bytes": artifacts.partial_video().stat().st_size,
                },
                "global_projection_jsonl": {
                    "path": str(artifacts.global_jsonl.resolve()),
                    "bytes": artifacts.partial_jsonl().stat().st_size,
                },
                "rally_results_jsonl": {
                    "path": str(artifacts.rally_results_jsonl.resolve()),
                    "bytes": (
                        artifacts.partial_rally_results_jsonl().stat().st_size
                    ),
                },
                "manifest": {
                    "path": str(artifacts.manifest.resolve()),
                },
            },
            summary={
                "processed_frames": processed,
                "source_fps": source.fps,
                "elapsed_seconds": round(elapsed, 3),
                "processing_fps": round(processed / max(elapsed, 1e-9), 3),
                "model_inference": False,
                "projection_only": projection_only,
                "bounce_candidate_frames": bounce_frames,
                "paddle_hit_candidate_frames": hit_frames,
                "out_of_bounds_bounce_candidate_frames": outside_bounce_frames,
                "second_bounce_candidate_frames": second_bounce_frames,
                "referee": (
                    referee_engine.diagnostics()
                    if referee_engine is not None
                    else {"enabled": False}
                ),
            },
        )
        write_manifest(artifacts.partial_manifest(), completed)
        artifacts.partial_video().replace(artifacts.video)
        artifacts.partial_jsonl().replace(artifacts.global_jsonl)
        artifacts.partial_rally_results_jsonl().replace(
            artifacts.rally_results_jsonl
        )
        artifacts.partial_manifest().replace(artifacts.manifest)
    except BaseException:
        _remove_failed_replay(artifacts)
        raise
    print(
        f"[done:{run_id}] replay_frames={processed}, "
        f"elapsed={elapsed:.1f}s, model_inference=false, "
        f"projection_only={str(projection_only).lower()}"
    )
    print(f"[video] {artifacts.video}")
    print(f"[jsonl] {artifacts.global_jsonl}")
    print(f"[jsonl] {artifacts.rally_results_jsonl}")
    print(f"[manifest] {artifacts.manifest}")
    return artifacts


def _remove_failed_replay(artifacts: ProjectionReplayArtifacts) -> None:
    """Remove only files created for a new replay attempt."""
    for path in artifacts.all_paths():
        path.unlink(missing_ok=True)
    try:
        artifacts.run_dir.rmdir()
    except OSError:
        pass


def _record_timestamp_s(
    global_record: dict,
    frame_index: int,
    source_fps: float,
) -> float:
    value = global_record.get("timestamp")
    if value is not None:
        try:
            timestamp_s = float(value)
        except (TypeError, ValueError):
            timestamp_s = float("nan")
        if math.isfinite(timestamp_s) and timestamp_s >= 0.0:
            return timestamp_s
    return int(frame_index) / float(source_fps)


def _declared_source_frames(
    manifest: dict,
    input_by_side: dict[str, dict],
) -> int | None:
    summary_value = manifest.get("summary", {}).get("processed_frames")
    input_values = [
        item.get("frames")
        for item in (input_by_side["left"], input_by_side["right"])
        if item.get("frames") is not None
    ]
    input_frames: int | None = None
    if input_values:
        parsed_inputs = [
            _non_negative_integer(value, "source input frames")
            for value in input_values
        ]
        if len(set(parsed_inputs)) != 1:
            raise ValueError(
                f"Source left/right declared frame mismatch: {parsed_inputs}"
            )
        input_frames = parsed_inputs[0]

    if summary_value is None:
        return input_frames
    processed_frames = _non_negative_integer(
        summary_value,
        "source summary.processed_frames",
    )
    if input_frames is not None and processed_frames > input_frames:
        raise ValueError(
            "Source processed_frames exceeds declared input frames: "
            f"processed={processed_frames}, input={input_frames}"
        )
    return processed_frames


def _non_negative_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if parsed < 0 or not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{label} must be a non-negative integer")
    return parsed


def _validate_source_jsonl_streams(
    left_jsonl: Path,
    right_jsonl: Path,
    global_jsonl: Path,
    *,
    expected_frames: int | None,
) -> int:
    """Reject truncated, desynchronized or non-contiguous replay sources."""
    paths = (left_jsonl, right_jsonl, global_jsonl)
    count = 0
    with (
        left_jsonl.open("r", encoding="utf-8") as left_handle,
        right_jsonl.open("r", encoding="utf-8") as right_handle,
        global_jsonl.open("r", encoding="utf-8") as global_handle,
    ):
        handles = (left_handle, right_handle, global_handle)
        while True:
            lines = [handle.readline() for handle in handles]
            ended = [not line for line in lines]
            if all(ended):
                break
            if any(ended):
                raise RuntimeError(
                    "Source left/right/global JSONL frame counts do not match"
                )
            for path, line in zip(paths, lines, strict=True):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid source JSON at {path}:{count + 1}: "
                        f"{error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Source JSONL row must be an object at {path}:{count + 1}"
                    )
                try:
                    frame_index = int(record["frame_index"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"Invalid source frame_index at {path}:{count + 1}"
                    ) from error
                if frame_index != count:
                    raise RuntimeError(
                        "Source frame indexes must be contiguous from zero: "
                        f"{path}:{count + 1} expected={count}, "
                        f"actual={frame_index}"
                    )
            count += 1

    if expected_frames is not None and count != expected_frames:
        raise RuntimeError(
            "Source JSONL count does not match completed manifest: "
            f"expected={expected_frames}, actual={count}"
        )
    return count


def _artifact_identity(
    path: Path,
    recorded_output,
    *,
    role: str,
) -> dict:
    size = path.stat().st_size
    recorded = recorded_output if isinstance(recorded_output, dict) else {}
    recorded_size = recorded.get("bytes")
    if recorded_size is not None and int(recorded_size) != size:
        raise RuntimeError(
            f"Source {role} size does not match manifest: "
            f"expected={recorded_size}, actual={size}"
        )
    digest = sha256_file(path)
    recorded_digest = recorded.get("sha256")
    if (
        recorded_digest is not None
        and str(recorded_digest).strip().lower() != digest
    ):
        raise RuntimeError(f"Source {role} SHA-256 does not match manifest")
    return {"bytes": size, "sha256": digest}


def _manifest_frame_size(item: dict, side: str) -> tuple[int, int]:
    width = int(item.get("width", 0))
    height = int(item.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError(f"Invalid {side} frame size in source manifest")
    return width, height


def _resolve_output_path(
    outputs: dict,
    name: str,
    fallback: Path,
) -> Path:
    value = outputs.get(name, {})
    path_value = value.get("path") if isinstance(value, dict) else None
    recorded = Path(path_value).resolve() if path_value else None
    local = fallback.resolve()
    recorded_exists = recorded is not None and recorded.is_file()
    local_exists = local.is_file()
    if recorded_exists and local_exists:
        assert recorded is not None
        try:
            same_file = recorded.samefile(local)
        except OSError:
            same_file = False
        if not same_file and not _same_artifact_content(recorded, local):
            raise ValueError(
                "Source manifest path conflicts with the local run artifact: "
                f"recorded={recorded}, local={local}"
            )
        return local
    if local_exists:
        return local
    if recorded_exists:
        assert recorded is not None
        return recorded
    return recorded or local


def _same_artifact_content(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    return sha256_file(first) == sha256_file(second)


def _selection_from_record(record: dict) -> GlobalBallSelection:
    track = _ball_track_from_record(record.get("track"))
    return GlobalBallSelection(
        global_track_id=int(record.get("global_track_id", 1)),
        active_side=record.get("active_side"),
        state=str(record.get("state", "no_ball")),
        track=track,
        local_track_id=record.get("local_track_id"),
        switched_side=bool(record.get("switched_side", False)),
        previous_side=record.get("previous_side"),
    )


def _ball_track_from_record(record: dict | None) -> BallTrack | None:
    if record is None:
        return None
    allowed = {item.name for item in fields(BallTrack)}
    return BallTrack(
        **{
            name: value
            for name, value in record.items()
            if name in allowed
        }
    )


def _validate_replay_outputs(
    video_path: Path,
    jsonl_path: Path,
    rally_results_jsonl_path: Path,
    *,
    expected_frames: int,
    expected_fps: float,
    expected_size: tuple[int, int],
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen replay video: {video_path}")
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"Replay frame mismatch: expected={expected_frames}, "
            f"actual={actual_frames}"
        )
    if abs(actual_fps - expected_fps) > 1e-3:
        raise RuntimeError(
            f"Replay FPS mismatch: expected={expected_fps}, "
            f"actual={actual_fps}"
        )
    if actual_size != expected_size:
        raise RuntimeError(
            f"Replay size mismatch: expected={expected_size}, "
            f"actual={actual_size}"
        )
    with jsonl_path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    if line_count != expected_frames:
        raise RuntimeError(
            f"Replay JSONL mismatch: expected={expected_frames}, "
            f"actual={line_count}"
        )
    return validate_rally_result_contract(
        jsonl_path,
        rally_results_jsonl_path,
    )
