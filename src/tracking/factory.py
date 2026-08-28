"""Configuration loading and tracking-component assembly.

Application entry points import this module instead of importing one another.
Keeping construction here makes the single-video, dual-camera, benchmark and
future board entry points share exactly the same component wiring.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .ball_detector import NullBallDetector, YoloBallDetector
from .ball_pipeline import BallTrackingPipeline
from .camera_motion import CameraMotionEstimator
from .fast_motion import FastMotionProposalGenerator
from .multi_ball_tracker import MultiBallTracker
from .onnx_detector import OnnxBallDetector
from .person_detector import YoloPersonDetector
from .person_tracking import PersonBoxTracker, PlayerSelector
from .temporal_motion import TemporalMotionFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("schema_version") != 1:
        raise ValueError("Config must declare schema_version: 1")
    return config


def build_detector(config: dict):
    detector_config = config.get("detector", {})
    model_value = detector_config.get("model")
    backend = str(detector_config.get("backend", "auto")).lower()
    if backend == "null" or model_value in (None, "null"):
        return NullBallDetector()

    model_path = project_path(model_value)
    if not model_path.exists():
        raise FileNotFoundError(f"Detector model does not exist: {model_path}")
    if backend == "auto":
        backend = (
            "onnxruntime"
            if model_path.suffix.lower() == ".onnx"
            else "ultralytics"
        )

    common = {
        "model_path": model_path,
        "ball_class_id": detector_config.get("ball_class_id", 0),
        "conf_threshold": detector_config.get("low_conf", 0.08),
        "iou_threshold": detector_config.get("iou_threshold", 0.5),
        "imgsz": detector_config.get("imgsz", 640),
        "max_detections": detector_config.get("max_detections", 32),
        "exclude_region": detector_config.get("exclude_region"),
    }
    if backend == "onnxruntime":
        return OnnxBallDetector(
            providers=detector_config.get("providers"),
            **common,
        )
    if backend == "ultralytics":
        return YoloBallDetector(
            device=detector_config.get("device"),
            **common,
        )
    raise ValueError(f"Unsupported detector backend: {backend}")


def build_tracker(
    config: dict,
    fps: float | None = None,
    frame_scale_override: float | None = None,
) -> MultiBallTracker:
    values = dict(config.get("tracker", {}))
    if fps is not None and fps > 0:
        values["default_fps"] = float(fps)
    if frame_scale_override is not None:
        values["frame_scale_override"] = float(frame_scale_override)
    return MultiBallTracker(**values)


def build_camera_motion_estimator(
    config: dict,
) -> CameraMotionEstimator | None:
    values = dict(config.get("runtime", {}).get("camera_motion", {}))
    enabled = bool(values.pop("enabled", False))
    return CameraMotionEstimator(**values) if enabled else None


def build_temporal_motion_filter(
    config: dict,
) -> TemporalMotionFilter | None:
    values = dict(config.get("runtime", {}).get("temporal_motion", {}))
    enabled = bool(values.pop("enabled", False))
    return TemporalMotionFilter(**values) if enabled else None


def build_fast_motion_proposal_generator(
    config: dict,
    frame_scale_override: float | None = None,
) -> FastMotionProposalGenerator | None:
    values = dict(config.get("runtime", {}).get("fast_motion", {}))
    enabled = bool(values.pop("enabled", False))
    if frame_scale_override is not None:
        values["frame_scale_override"] = float(frame_scale_override)
    return FastMotionProposalGenerator(**values) if enabled else None


def build_person_detector(config: dict) -> YoloPersonDetector | None:
    values = dict(config.get("runtime", {}).get("person_detection", {}))
    enabled = bool(values.pop("enabled", False))
    if not enabled:
        return None
    model_value = values.pop("model", None)
    if not model_value:
        raise ValueError(
            "runtime.person_detection.model is required when enabled"
        )
    model_path = project_path(model_value)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Person detector model does not exist: {model_path}. "
            "Place a COCO person model there before running this branch."
        )
    values.pop("interval_frames", None)
    values.pop("fail_on_error", None)
    values.pop("box_tracker", None)
    values.pop("player_selection", None)
    return YoloPersonDetector(model_path=model_path, **values)


def build_person_tracking(
    config: dict,
    fps: float,
) -> tuple[PersonBoxTracker | None, PlayerSelector | None]:
    person_config = config.get("runtime", {}).get("person_detection", {})
    if not person_config.get("enabled", False):
        return None, None
    tracker_values = dict(person_config.get("box_tracker", {}))
    tracker_values["default_fps"] = fps
    selector_values = dict(person_config.get("player_selection", {}))
    return PersonBoxTracker(**tracker_values), PlayerSelector(**selector_values)


def build_pipeline(
    config: dict,
    fps: float,
    detector,
    person_detector=None,
    *,
    frame_scale_override: float | None = None,
) -> BallTrackingPipeline:
    person_tracker, player_selector = build_person_tracking(config, fps)
    person_config = config.get("runtime", {}).get("person_detection", {})
    detector_config = config.get("detector", {})
    return BallTrackingPipeline(
        detector,
        build_tracker(
            config,
            fps=fps,
            frame_scale_override=frame_scale_override,
        ),
        detector_interval=config.get("runtime", {}).get("detector_interval", 1),
        duplicate_iou_threshold=detector_config.get(
            "duplicate_iou_threshold",
            0.20,
        ),
        duplicate_center_scale=detector_config.get(
            "duplicate_center_scale",
            0.75,
        ),
        duplicate_center_px=detector_config.get(
            "duplicate_center_px",
            6.0,
        ),
        camera_motion_estimator=build_camera_motion_estimator(config),
        temporal_motion_filter=build_temporal_motion_filter(config),
        fast_motion_proposal_generator=build_fast_motion_proposal_generator(
            config,
            frame_scale_override=frame_scale_override,
        ),
        person_detector=person_detector,
        person_tracker=person_tracker,
        player_selector=player_selector,
        person_detector_interval=person_config.get("interval_frames", 5),
    )
