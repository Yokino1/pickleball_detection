"""Synchronized dual-camera processing loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2

from ...court import (
    FixedCourtProjector,
    ProjectionResult,
    build_court_projection,
)
from ...referee import REFEREE_CONTRACT_VERSION, build_referee_engine
from ..factory import PROJECT_ROOT, build_pipeline
from ..overlay import TrackingOverlay
from ..run_manifest import (
    complete_manifest,
    create_manifest,
    write_manifest,
)
from ..types import CourtInfo
from .artifacts import DualRunArtifacts, validate_written_outputs
from .coordinator import CrossCameraHandoffAdvisor, CrossHalfBallCoordinator
from .rendering import (
    HEADER_HEIGHT,
    display_result,
    make_header,
)
from .rendering import (
    court_motion_discontinuity_reason as _court_motion_discontinuity_reason,
)


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
    frame_widths = {
        "left": left_width,
        "right": right_width,
    }
    frame_scale_mode = _stream_frame_scale_mode(config)
    frame_scale_overrides = _stream_frame_scale_overrides(
        config,
        frame_widths,
    )
    (
        court_projector,
        court_renderer,
        court_event_interpreter,
    ) = build_court_projection(
        config,
        {
            "left": (left_width, left_height),
            "right": (right_width, right_height),
        },
    )
    referee_engine = build_referee_engine(
        config,
        court_projector.layout if court_projector is not None else None,
    )
    court_panel_width = (
        court_renderer.panel_width(left_height)
        if court_renderer is not None
        else 0
    )

    artifacts = DualRunArtifacts.for_run(output_dir, run_id)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
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
            "frame_scale_mode": frame_scale_mode,
            "frame_scale_overrides": frame_scale_overrides,
            "header_height": HEADER_HEIGHT,
            "court_projection_enabled": court_projector is not None,
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
            "court_panel_width": court_panel_width,
            "run_note": run_note,
        },
    )
    write_manifest(partial.manifest, manifest)

    output_config = config.get("output", {})
    fourcc = cv2.VideoWriter_fourcc(*output_config.get("codec", "mp4v"))
    output_size = (
        left_width + right_width + court_panel_width,
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
    handoff_config = config.get("runtime", {}).get(
        "dual_camera_handoff",
        {},
    )
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
        strict_handoff=handoff_config.get(
            "strict_state_machine",
            False,
        ),
        receiver_confirmation_hits=handoff_config.get(
            "receiver_confirmation_hits",
            2,
        ),
        switch_lock_ms=handoff_config.get("switch_lock_ms", 100.0),
        left_net_edge=handoff_config.get("left_net_edge", "right"),
        right_net_edge=handoff_config.get("right_net_edge", "left"),
        observation_first=handoff_config.get("observation_first", False),
        primary_observation_sources=handoff_config.get(
            "primary_observation_sources",
        ),
    )
    handoff_advisor = _build_handoff_advisor(
        config,
        frame_scale_overrides,
    )

    processed = 0
    roi_retry_frames = 0
    fast_motion_proposal_frames = 0
    trail_reset_frames = 0
    projected_frames = 0
    projected_predicted_frames = 0
    projected_outside_frames = 0
    bounce_candidate_frames = 0
    out_of_bounds_bounce_candidate_frames = 0
    second_bounce_candidate_frames = 0
    paddle_hit_candidate_frames = 0
    previous_rendered_tracks: dict[
        str,
        tuple[int | None, tuple[float, float], float] | None,
    ] = {
        "left": None,
        "right": None,
    }
    started = time.perf_counter()
    completed = False
    try:
        with (
            partial.left_jsonl.open("w", encoding="utf-8") as left_handle,
            partial.right_jsonl.open("w", encoding="utf-8") as right_handle,
            partial.global_jsonl.open("w", encoding="utf-8") as global_handle,
            partial.rally_results_jsonl.open(
                "w",
                encoding="utf-8",
            ) as rally_results_handle,
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

                handoff_status = _handoff_status(
                    handoff_advisor,
                    timestamp_s,
                    {
                        "left": left_search_roi,
                        "right": right_search_roi,
                    },
                )
                selection = coordinator.update(
                    left_result.ball_tracks,
                    right_result.ball_tracks,
                    timestamp_s=timestamp_s,
                    frame_widths={
                        "left": left_width,
                        "right": right_width,
                    },
                    handoff=handoff_status,
                )
                court_projection = (
                    court_projector.project_track(
                        selection.active_side,
                        selection.track,
                    )
                    if court_projector is not None
                    else None
                )
                trail_reset_reason = _trail_reset_reason(
                    selection,
                    timestamp_s=timestamp_s,
                    previous_rendered_tracks=previous_rendered_tracks,
                    max_speed_px_per_second=tracker_config.get(
                        "max_speed_px_per_second",
                        0.0,
                    ),
                    frame_scale_overrides=frame_scale_overrides,
                    max_observation_gap_s=max(1e-6, 1.5 / left_fps),
                )
                court_discontinuity_reason = _court_motion_discontinuity_reason(
                    trail_reset_reason
                )
                if trail_reset_reason is not None:
                    trail_reset_frames += 1
                    if selection.switched_side:
                        left_overlay.reset()
                        right_overlay.reset()
                    elif selection.active_side == "left":
                        left_overlay.reset()
                    elif selection.active_side == "right":
                        right_overlay.reset()
                    if (
                        court_renderer is not None
                        and trail_reset_reason != "camera_side_switch"
                    ):
                        court_renderer.reset()
                active_result = None
                if selection.active_side == "left":
                    active_result = left_result
                elif selection.active_side == "right":
                    active_result = right_result
                eligible_players = (
                    active_result.players if active_result is not None else []
                )
                eligible_player_centers_court_xy = (
                    court_projector.project_eligible_player_box_centers(
                        selection.active_side,
                        eligible_players,
                    )
                    if court_projector is not None
                    else []
                )
                if (
                    court_event_interpreter is not None
                    and court_projection is not None
                ):
                    court_projection.event = court_event_interpreter.update(
                        court_projection,
                        track=selection.track,
                        active_side=selection.active_side,
                        local_track_id=selection.local_track_id,
                        tracker_diagnostics={
                            "left": left_result.diagnostics.get("tracker", {}),
                            "right": right_result.diagnostics.get("tracker", {}),
                        },
                        frame_scale_overrides=frame_scale_overrides,
                        discontinuity_reason=court_discontinuity_reason,
                        timestamp_s=timestamp_s,
                        frame_index=processed,
                        eligible_players=eligible_players,
                    ).to_dict()
                referee_result = (
                    referee_engine.update(
                        court_projection,
                        timestamp_s=timestamp_s,
                        frame_index=processed,
                        discontinuity_reason=court_discontinuity_reason,
                        eligible_player_centers_court_xy=(
                            eligible_player_centers_court_xy
                        ),
                    )
                    if referee_engine is not None
                    and court_projection is not None
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
                if handoff_advisor is not None:
                    handoff_advisor.update(
                        selection,
                        timestamp_s=timestamp_s,
                        frame_widths={
                            "left": left_width,
                            "right": right_width,
                        },
                    )
                if court_projection is not None and court_projector is not None:
                    _attach_court_info(
                        left_result,
                        right_result,
                        court_projection,
                        court_projector,
                    )
                    projected_frames += int(court_projection.projection_valid)
                    projected_predicted_frames += int(
                        court_projection.projection_valid
                        and court_projection.predicted
                    )
                    projected_outside_frames += int(
                        court_projection.projection_valid
                        and court_projection.inside_court is False
                    )
                    event_names = set(
                        (court_projection.event or {}).get("events", [])
                    )
                    bounce_candidate_frames += int(
                        "bounce_candidate" in event_names
                    )
                    out_of_bounds_bounce_candidate_frames += int(
                        "out_of_bounds_bounce_candidate" in event_names
                    )
                    second_bounce_candidate_frames += int(
                        "second_bounce_candidate" in event_names
                    )
                    paddle_hit_candidate_frames += int(
                        "paddle_hit_candidate" in event_names
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
                            trail_reset_reason=trail_reset_reason,
                            court_projection=court_projection,
                            referee=referee_record,
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
                _remember_rendered_track(
                    selection,
                    timestamp_s,
                    previous_rendered_tracks,
                )
                rendered_parts = [left_rendered, right_rendered]
                if court_renderer is not None and court_projection is not None:
                    rendered_parts.append(
                        court_renderer.render(
                            left_rendered.shape[0],
                            court_projection,
                            referee=referee_record,
                        )
                    )
                combined = cv2.hconcat(rendered_parts)
                combined = cv2.vconcat(
                    [
                        make_header(
                            combined.shape[1],
                            left_rendered.shape[1],
                            selection,
                            court_panel_width=court_panel_width,
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
                if court_panel_width > 0:
                    court_divider_x = (
                        left_rendered.shape[1] + right_rendered.shape[1]
                    )
                    cv2.line(
                        combined,
                        (court_divider_x, 0),
                        (court_divider_x, combined.shape[0] - 1),
                        (255, 160, 0),
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
                "trail_reset_frames": trail_reset_frames,
                "projected_frames": projected_frames,
                "projected_predicted_frames": projected_predicted_frames,
                "projected_outside_frames": projected_outside_frames,
                "bounce_candidate_frames": bounce_candidate_frames,
                "out_of_bounds_bounce_candidate_frames": (
                    out_of_bounds_bounce_candidate_frames
                ),
                "second_bounce_candidate_frames": (
                    second_bounce_candidate_frames
                ),
                "paddle_hit_candidate_frames": paddle_hit_candidate_frames,
                "referee": (
                    referee_engine.diagnostics()
                    if referee_engine is not None
                    else {"enabled": False}
                ),
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
    print(f"[jsonl] {artifacts.rally_results_jsonl}")
    print(f"[manifest] {artifacts.manifest}")
    return processed


def _court_info(
    projection: ProjectionResult,
    projector: FixedCourtProjector,
    side: str,
) -> CourtInfo:
    calibration = projector.calibration_for(side)
    is_active = side == projection.active_side
    if is_active:
        return CourtInfo(
            coordinate_system=projection.coordinate_system,
            coordinate_system_version=projection.coordinate_system_version,
            active_side=projection.active_side,
            calibration_id=projection.calibration_id,
            calibration_source=projection.calibration_source,
            image_xy=projection.image_xy,
            projection_status=projection.projection_status,
            projection_valid=projection.projection_valid,
            homography_available=projection.homography_available,
            ball_court_xy=projection.ball_court_xy,
            reprojection_error_px=projection.reprojection_error_px,
            track_status=projection.track_status,
            observed=projection.observed,
            predicted=projection.predicted,
            inside_court=projection.inside_court,
            event=projection.event,
            projection_warnings=projection.projection_warnings,
        )
    return CourtInfo(
        coordinate_system=projector.layout.coordinate_system,
        coordinate_system_version=projector.layout.coordinate_system_version,
        active_side=projection.active_side,
        calibration_id=(
            calibration.calibration_id if calibration is not None else None
        ),
        calibration_source=(
            calibration.calibration_source
            if calibration is not None
            else None
        ),
        projection_status="none",
        projection_valid=False,
        homography_available=bool(
            calibration and calibration.homography_available
        ),
        projection_warnings=(
            list(calibration.warnings)
            if calibration is not None
            else ["calibration_unavailable"]
        ),
    )


def _attach_court_info(
    left_result,
    right_result,
    projection: ProjectionResult,
    projector: FixedCourtProjector,
) -> None:
    left_result.court = _court_info(projection, projector, "left")
    right_result.court = _court_info(projection, projector, "right")


def _read_and_validate_metadata(left_capture, right_capture) -> dict:
    def read(capture, side: str) -> dict:
        raw = {
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frames": float(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        for name, value in raw.items():
            valid_range = value > 0.0 if name == "fps" else value >= 1.0
            if not math.isfinite(value) or not valid_range:
                label = "FPS" if name == "fps" else f"video {name}"
                raise ValueError(f"Invalid {side} {label}: {value}")
        return {
            "fps": raw["fps"],
            "frames": int(raw["frames"]),
            "width": int(raw["width"]),
            "height": int(raw["height"]),
        }

    left = read(left_capture, "left")
    right = read(right_capture, "right")
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
    values.pop("strict_state_machine", None)
    values.pop("receiver_confirmation_hits", None)
    values.pop("switch_lock_ms", None)
    values.pop("observation_first", None)
    values.pop("primary_observation_sources", None)
    values["frame_scale_overrides"] = frame_scale_overrides
    return CrossCameraHandoffAdvisor(**values) if enabled else None


def _stream_frame_scale_mode(config: dict) -> str:
    stream_config = config.get("runtime", {}).get(
        "dual_camera_streams",
        {},
    )
    mode = str(
        stream_config.get("frame_scale_mode", "per_stream_width")
    ).strip()
    valid = {"per_stream_width", "paired_crop_total_width"}
    if mode not in valid:
        raise ValueError(
            "runtime.dual_camera_streams.frame_scale_mode must be one of "
            f"{sorted(valid)}, found {mode!r}"
        )
    return mode


def _stream_frame_scale_overrides(
    config: dict,
    frame_widths: dict[str, int],
) -> dict[str, float]:
    stream_config = config.get("runtime", {}).get(
        "dual_camera_streams",
        {},
    )
    mode = _stream_frame_scale_mode(config)
    overrides: dict[str, float] = {}
    if mode == "paired_crop_total_width":
        reference_width = float(
            config.get("tracker", {}).get(
                "reference_frame_width",
                1280.0,
            )
        )
        if reference_width <= 0.0:
            raise ValueError("tracker.reference_frame_width must be positive")
        pair_width = sum(
            max(0, int(frame_widths.get(side, 0)))
            for side in ("left", "right")
        )
        if pair_width <= 0:
            raise ValueError(
                "paired_crop_total_width requires positive left/right widths"
            )
        inferred = max(0.25, pair_width / reference_width)
        overrides = {"left": inferred, "right": inferred}

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


def _trail_reset_reason(
    selection,
    *,
    timestamp_s: float,
    previous_rendered_tracks: dict[
        str,
        tuple[int | None, tuple[float, float], float] | None,
    ],
    max_speed_px_per_second: float,
    frame_scale_overrides: dict[str, float],
    max_observation_gap_s: float | None = None,
) -> str | None:
    track = selection.track
    side = selection.active_side
    if track is None or side not in ("left", "right") or track.center is None:
        return None
    if selection.switched_side:
        return "camera_side_switch"
    previous = previous_rendered_tracks.get(side)
    if previous is None:
        return None
    previous_track_id, previous_center, previous_timestamp_s = previous
    if previous_track_id != selection.local_track_id:
        return "local_track_change"
    elapsed_s = float(timestamp_s) - previous_timestamp_s
    if (
        max_observation_gap_s is not None
        and max_observation_gap_s > 0.0
        and elapsed_s > max_observation_gap_s
    ):
        return "observation_gap"
    if elapsed_s <= 0.0 or max_speed_px_per_second <= 0.0:
        return None
    distance = float(
        (
            (float(track.center[0]) - previous_center[0]) ** 2
            + (float(track.center[1]) - previous_center[1]) ** 2
        )
        ** 0.5
    )
    scale = float(frame_scale_overrides.get(side, 1.0))
    if distance / elapsed_s > max_speed_px_per_second * scale:
        return "physical_discontinuity"
    return None


def _remember_rendered_track(
    selection,
    timestamp_s: float,
    previous_rendered_tracks: dict[
        str,
        tuple[int | None, tuple[float, float], float] | None,
    ],
) -> None:
    track = selection.track
    side = selection.active_side
    if track is None or side not in ("left", "right") or track.center is None:
        return
    previous_rendered_tracks[side] = (
        selection.local_track_id,
        (float(track.center[0]), float(track.center[1])),
        float(timestamp_s),
    )


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


def _handoff_status(
    advisor: CrossCameraHandoffAdvisor | None,
    timestamp_s: float,
    search_rois: dict[str, list[float] | None],
) -> dict | None:
    if advisor is None:
        return None
    status = advisor.diagnostics(timestamp_s)
    target_side = status.get("target_side")
    status["target_roi"] = (
        search_rois.get(target_side)
        if target_side in ("left", "right")
        else None
    )
    return status


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
    *,
    trail_reset_reason: str | None = None,
    court_projection: ProjectionResult | None = None,
    referee: dict | None = None,
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
        "court": (
            court_projection.to_dict()
            if court_projection is not None
            else None
        ),
        "referee": referee,
        "coordinator": coordinator.diagnostics(),
        "handoff": (
            handoff_advisor.diagnostics(timestamp_s)
            if handoff_advisor is not None
            else {"active": False, "enabled": False}
        ),
        "rendering": {
            "trail_reset": trail_reset_reason is not None,
            "trail_reset_reason": trail_reset_reason,
        },
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
        "rally_results_jsonl",
    ):
        final_path = getattr(final, name)
        partial_path = getattr(partial, name)
        records[name] = {
            "path": str(final_path.resolve()),
            "bytes": partial_path.stat().st_size,
        }
    records["manifest"] = {"path": str(final.manifest.resolve())}
    return records
