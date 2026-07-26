"""Synchronized dual-camera processing loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from ..factory import PROJECT_ROOT, build_pipeline
from ..overlay import TrackingOverlay
from ..run_manifest import (
    complete_manifest,
    create_manifest,
    write_manifest,
)
from .artifacts import DualRunArtifacts, validate_written_outputs
from .coordinator import CrossCameraHandoffAdvisor, CrossHalfBallCoordinator
from .rendering import HEADER_HEIGHT, display_result, make_header


def run_pair(
    run_id: str,
    left_path: Path,
    right_path: Path,
    output_dir: Path,
    config: dict,
    detector,
    person_detector,
    *,
    config_path: Path,
    max_frames: int | None = None,
    global_switch_min_missing_ms: float = 20.0,
    run_note: str | None = None,
) -> int:
    left_capture = cv2.VideoCapture(str(left_path))
    right_capture = cv2.VideoCapture(str(right_path))
    if not left_capture.isOpened() or not right_capture.isOpened():
        left_capture.release()
        right_capture.release()
        raise RuntimeError(
            f"Cannot open synchronized pair: {left_path}, {right_path}"
        )

    try:
        metadata = _read_and_validate_metadata(left_capture, right_capture)
    except BaseException:
        left_capture.release()
        right_capture.release()
        raise
    left_fps = metadata["left"]["fps"]
    left_frames = metadata["left"]["frames"]
    left_width = metadata["left"]["width"]
    right_width = metadata["right"]["width"]
    left_height = metadata["left"]["height"]
    right_height = metadata["right"]["height"]
    frame_scale_overrides = _stream_frame_scale_overrides(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = DualRunArtifacts.for_run(output_dir, run_id)
    partial = artifacts.partial()
    partial.remove()

    manifest = create_manifest(
        project_root=PROJECT_ROOT,
        run_id=run_id,
        run_type="dual_camera_tracking",
        config_path=config_path,
        config=config,
        inputs=[
            {"side": "left", "path": str(left_path.resolve()), **metadata["left"]},
            {
                "side": "right",
                "path": str(right_path.resolve()),
                **metadata["right"],
            },
        ],
        parameters={
            "max_frames": max_frames,
            "global_switch_min_missing_ms": global_switch_min_missing_ms,
            "frame_scale_overrides": frame_scale_overrides,
            "header_height": HEADER_HEIGHT,
            "run_note": run_note,
        },
    )
    write_manifest(partial.manifest, manifest)

    output_config = config.get("output", {})
    fourcc = cv2.VideoWriter_fourcc(*output_config.get("codec", "mp4v"))
    output_size = (
        left_width + right_width,
        left_height + HEADER_HEIGHT,
    )
    writer = cv2.VideoWriter(
        str(partial.video),
        fourcc,
        left_fps,
        output_size,
    )
    if not writer.isOpened():
        left_capture.release()
        right_capture.release()
        partial.remove()
        raise RuntimeError(f"Cannot create output video: {artifacts.video}")

    left_pipeline = build_pipeline(
        config,
        left_fps,
        detector,
        person_detector,
        frame_scale_override=frame_scale_overrides.get("left"),
    )
    right_pipeline = build_pipeline(
        config,
        metadata["right"]["fps"],
        detector,
        person_detector,
        frame_scale_override=frame_scale_overrides.get("right"),
    )
    left_overlay = _build_overlay(output_config)
    right_overlay = _build_overlay(output_config)
    person_config = config.get("runtime", {}).get("person_detection", {})
    tracker_config = config.get("tracker", {})
    coordinator = CrossHalfBallCoordinator(
        switch_min_missing_ms=global_switch_min_missing_ms,
        max_continuity_speed_px_per_second=tracker_config.get(
            "max_speed_px_per_second",
            0.0,
        ),
        reference_frame_width=tracker_config.get(
            "reference_frame_width",
            1280.0,
        ),
        frame_scale_overrides=frame_scale_overrides,
    )
    handoff_advisor = _build_handoff_advisor(
        config,
        frame_scale_overrides,
    )

    processed = 0
    roi_retry_frames = 0
    fast_motion_proposal_frames = 0
    started = time.perf_counter()
    completed = False
    try:
        with (
            partial.left_jsonl.open("w", encoding="utf-8") as left_handle,
            partial.right_jsonl.open("w", encoding="utf-8") as right_handle,
            partial.global_jsonl.open("w", encoding="utf-8") as global_handle,
        ):
            while max_frames is None or processed < max_frames:
                left_ok, left_frame = left_capture.read()
                right_ok, right_frame = right_capture.read()
                if left_ok != right_ok:
                    raise RuntimeError(
                        f"Pair desynchronized at frame {processed}"
                    )
                if not left_ok:
                    break

                timestamp_s = processed / left_fps
                left_search_roi = _search_roi(
                    handoff_advisor,
                    "left",
                    timestamp_s,
                    left_width,
                    left_height,
                )
                right_search_roi = _search_roi(
                    handoff_advisor,
                    "right",
                    timestamp_s,
                    right_width,
                    right_height,
                )
                left_result = left_pipeline.process_frame(
                    left_frame,
                    processed,
                    timestamp_s,
                    search_roi=left_search_roi,
                )
                right_result = right_pipeline.process_frame(
                    right_frame,
                    processed,
                    timestamp_s,
                    search_roi=right_search_roi,
                )
                _raise_for_detector_errors(
                    config,
                    person_config,
                    person_detector,
                    left_result,
                    right_result,
                )

                selection = coordinator.update(
                    left_result.ball_tracks,
                    right_result.ball_tracks,
                    timestamp_s=timestamp_s,
                    frame_widths={
                        "left": left_width,
                        "right": right_width,
                    },
                )
                if handoff_advisor is not None:
                    handoff_advisor.update(
                        selection,
                        timestamp_s=timestamp_s,
                        frame_widths={
                            "left": left_width,
                            "right": right_width,
                        },
                    )

                left_handle.write(
                    json.dumps(left_result.to_dict(), ensure_ascii=False) + "\n"
                )
                right_handle.write(
                    json.dumps(right_result.to_dict(), ensure_ascii=False) + "\n"
                )
                global_handle.write(
                    json.dumps(
                        _global_frame_record(
                            processed,
                            timestamp_s,
                            selection,
                            coordinator,
                            handoff_advisor,
                            left_result,
                            right_result,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                roi_retry_frames += int(
                    left_result.diagnostics["roi_retry_used"]
                    or right_result.diagnostics["roi_retry_used"]
                )
                fast_motion_proposal_frames += int(
                    left_result.diagnostics["fast_motion_proposal_count"] > 0
                    or right_result.diagnostics["fast_motion_proposal_count"] > 0
                )

                left_rendered = left_overlay.draw(
                    left_frame.copy(),
                    display_result(
                        left_result,
                        selection.track
                        if selection.active_side == "left"
                        else None,
                    ),
                )
                right_rendered = right_overlay.draw(
                    right_frame.copy(),
                    display_result(
                        right_result,
                        selection.track
                        if selection.active_side == "right"
                        else None,
                    ),
                )
                combined = cv2.hconcat([left_rendered, right_rendered])
                combined = cv2.vconcat(
                    [
                        make_header(
                            combined.shape[1],
                            left_rendered.shape[1],
                            selection,
                        ),
                        combined,
                    ]
                )
                divider_x = left_rendered.shape[1]
                cv2.line(
                    combined,
                    (divider_x, 0),
                    (divider_x, combined.shape[0] - 1),
                    (0, 255, 0),
                    2,
                )
                writer.write(combined)
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{run_id}] {processed}/{left_frames} frames "
                        f"({processed / max(elapsed, 1e-6):.2f} fps)"
                    )

        expected = (
            left_frames
            if max_frames is None
            else min(left_frames, max_frames)
        )
        if processed != expected:
            raise RuntimeError(
                f"Output frame mismatch: expected={expected}, wrote={processed}"
            )
        completed = True
    finally:
        left_capture.release()
        right_capture.release()
        writer.release()
        if not completed:
            partial.remove()

    elapsed = time.perf_counter() - started
    try:
        validate_written_outputs(
            partial,
            expected_frames=expected,
            expected_fps=left_fps,
            expected_size=output_size,
        )
        manifest = complete_manifest(
            manifest,
            outputs=_manifest_outputs(artifacts, partial),
            summary={
                "processed_frames": processed,
                "source_fps": left_fps,
                "elapsed_seconds": round(elapsed, 3),
                "processing_fps": round(
                    processed / max(elapsed, 1e-6),
                    3,
                ),
                "handoff_count": coordinator.handoff_count,
                "handoff_trigger_count": (
                    handoff_advisor.trigger_count
                    if handoff_advisor is not None
                    else 0
                ),
                "roi_retry_frames": roi_retry_frames,
                "fast_motion_proposal_frames": fast_motion_proposal_frames,
            },
        )
        write_manifest(partial.manifest, manifest)
        artifacts.promote_from(partial)
    except BaseException:
        partial.remove()
        raise

    print(
        f"[done:{run_id}] frames={processed}, fps={left_fps:.3f}, "
        f"elapsed={elapsed:.1f}s"
    )
    print(f"[video] {artifacts.video}")
    print(f"[jsonl] {artifacts.left_jsonl}")
    print(f"[jsonl] {artifacts.right_jsonl}")
    print(f"[jsonl] {artifacts.global_jsonl}")
    print(f"[manifest] {artifacts.manifest}")
    return processed


def _read_and_validate_metadata(left_capture, right_capture) -> dict:
    def read(capture) -> dict:
        return {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)) or 30.0,
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    left = read(left_capture)
    right = read(right_capture)
    if abs(left["fps"] - right["fps"]) > 1e-3:
        raise ValueError(f"FPS mismatch: left={left['fps']}, right={right['fps']}")
    if left["frames"] != right["frames"]:
        raise ValueError(
            "Frame-count mismatch: "
            f"left={left['frames']}, right={right['frames']}"
        )
    if left["height"] != right["height"]:
        raise ValueError(
            "Frame-height mismatch: "
            f"left={left['height']}, right={right['height']}"
        )
    return {"left": left, "right": right}


def _build_overlay(output_config: dict) -> TrackingOverlay:
    return TrackingOverlay(
        trail_length=output_config.get("trail_length", 10),
        draw_raw_detections=output_config.get(
            "draw_raw_detections",
            False,
        ),
        draw_players=output_config.get("draw_players", True),
    )


def _build_handoff_advisor(
    config: dict,
    frame_scale_overrides: dict[str, float],
) -> CrossCameraHandoffAdvisor | None:
    values = dict(config.get("runtime", {}).get("dual_camera_handoff", {}))
    enabled = bool(values.pop("enabled", True))
    values["frame_scale_overrides"] = frame_scale_overrides
    return CrossCameraHandoffAdvisor(**values) if enabled else None


def _stream_frame_scale_overrides(config: dict) -> dict[str, float]:
    stream_config = config.get("runtime", {}).get(
        "dual_camera_streams",
        {},
    )
    overrides = {}
    for side in ("left", "right"):
        value = stream_config.get(side, {}).get("frame_scale_override")
        if value is None:
            continue
        scale = float(value)
        if scale < 0.25:
            raise ValueError(
                f"runtime.dual_camera_streams.{side}."
                "frame_scale_override must be at least 0.25"
            )
        overrides[side] = scale
    return overrides


def _search_roi(
    advisor: CrossCameraHandoffAdvisor | None,
    side: str,
    timestamp_s: float,
    width: int,
    height: int,
):
    if advisor is None:
        return None
    return advisor.search_roi(
        side,
        timestamp_s=timestamp_s,
        frame_width=width,
        frame_height=height,
    )


def _raise_for_detector_errors(
    config: dict,
    person_config: dict,
    person_detector,
    left_result,
    right_result,
) -> None:
    for side, result in (("left", left_result), ("right", right_result)):
        if result.diagnostics["detector_error"] and config.get(
            "runtime",
            {},
        ).get("fail_on_detector_error", True):
            raise RuntimeError(f"Ball detector failed on {side} frame")
        if result.diagnostics["person_detector_error"] and person_config.get(
            "fail_on_error",
            True,
        ):
            error = getattr(person_detector, "last_error", "unknown error")
            raise RuntimeError(
                f"Person detector failed on {side} frame: {error}"
            )


def _global_frame_record(
    frame_index,
    timestamp_s,
    selection,
    coordinator,
    handoff_advisor,
    left_result,
    right_result,
) -> dict:
    def side_record(result) -> dict:
        return {
            "candidate_count": len(result.ball_tracks),
            "candidate_track_ids": [
                track.track_id for track in result.ball_tracks
            ],
            "eligible_player_count": sum(
                person.eligible_player for person in result.players
            ),
        }

    return {
        "frame_index": frame_index,
        "timestamp": timestamp_s,
        "global_ball": selection.to_dict(),
        "coordinator": coordinator.diagnostics(),
        "handoff": (
            handoff_advisor.diagnostics(timestamp_s)
            if handoff_advisor is not None
            else {"active": False, "enabled": False}
        ),
        "left": side_record(left_result),
        "right": side_record(right_result),
    }


def _manifest_outputs(
    final: DualRunArtifacts,
    partial: DualRunArtifacts,
) -> dict:
    records = {}
    for name in (
        "video",
        "left_jsonl",
        "right_jsonl",
        "global_jsonl",
    ):
        final_path = getattr(final, name)
        partial_path = getattr(partial, name)
        records[name] = {
            "path": str(final_path.resolve()),
            "bytes": partial_path.stat().st_size,
        }
    records["manifest"] = {"path": str(final.manifest.resolve())}
    return records
