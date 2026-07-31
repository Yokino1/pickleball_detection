"""Load and validate fixed-camera manual court calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .layout import CourtLayout


@dataclass
class CameraCalibration:
    """One camera's fixed image-to-court homography and audit metadata."""

    side: str
    calibration_id: str | None
    calibration_source: str
    image_size: tuple[int, int]
    homography_image_to_court: np.ndarray | None
    reprojection_error_px: float | None
    source_keypoint_count: int
    valid: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def homography_available(self) -> bool:
        return self.valid and self.homography_image_to_court is not None


def _point_id(value: str) -> int | None:
    text = str(value)
    if not text.startswith("kp"):
        return None
    try:
        return int(text[2:])
    except ValueError:
        return None


def _invalid_calibration(
    side: str,
    values: dict,
    warnings: list[str],
    source_keypoint_count: int = 0,
) -> CameraCalibration:
    size = values.get("image_size", [0, 0])
    return CameraCalibration(
        side=side,
        calibration_id=values.get("calibration_id"),
        calibration_source=str(
            values.get("source", "manual_rough_fixed_view")
        ),
        image_size=(int(size[0]), int(size[1])),
        homography_image_to_court=None,
        reprojection_error_px=None,
        source_keypoint_count=source_keypoint_count,
        valid=False,
        warnings=warnings,
    )


def _load_camera_calibration(
    side: str,
    values: dict,
    layout: CourtLayout,
    actual_image_size: tuple[int, int],
) -> CameraCalibration:
    warnings = [
        "ground_plane_homography_only",
        "airborne_ball_is_line_of_sight_ground_plane_approximation",
    ]
    configured_size = tuple(int(value) for value in values.get("image_size", []))
    if len(configured_size) != 2:
        return _invalid_calibration(
            side,
            values,
            [*warnings, "invalid_calibration_image_size"],
        )
    if configured_size != actual_image_size:
        return _invalid_calibration(
            side,
            values,
            [
                *warnings,
                "calibration_image_size_mismatch:"
                f"expected={configured_size},actual={actual_image_size}",
            ],
        )

    image_points = []
    court_points = []
    for point_name, image_xy in values.get("keypoints", {}).items():
        index = _point_id(point_name)
        if index not in layout.canonical_keypoints:
            continue
        if not isinstance(image_xy, (list, tuple)) or len(image_xy) != 2:
            continue
        image_point = np.asarray(image_xy, dtype=np.float64)
        if not np.all(np.isfinite(image_point)):
            continue
        image_points.append(image_point.tolist())
        court_points.append(layout.canonical_keypoints[index])

    point_count = len(image_points)
    if point_count < 4:
        return _invalid_calibration(
            side,
            values,
            [*warnings, f"too_few_calibration_points:{point_count}"],
            point_count,
        )

    image_array = np.asarray(image_points, dtype=np.float64)
    court_array = np.asarray(court_points, dtype=np.float64)
    image_area = float(
        cv2.contourArea(cv2.convexHull(image_array.astype(np.float32)))
    )
    court_area = float(
        cv2.contourArea(cv2.convexHull(court_array.astype(np.float32)))
    )
    if image_area <= 1e-3 or court_area <= 1e-6:
        return _invalid_calibration(
            side,
            values,
            [*warnings, "degenerate_calibration_points"],
            point_count,
        )

    homography, _mask = cv2.findHomography(image_array, court_array, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return _invalid_calibration(
            side,
            values,
            [*warnings, "homography_estimation_failed"],
            point_count,
        )
    try:
        court_to_image = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return _invalid_calibration(
            side,
            values,
            [*warnings, "homography_not_invertible"],
            point_count,
        )

    reprojected = cv2.perspectiveTransform(
        court_array.astype(np.float32).reshape(-1, 1, 2),
        court_to_image,
    ).reshape(-1, 2)
    errors = np.linalg.norm(
        reprojected.astype(np.float64) - image_array,
        axis=1,
    )
    mean_error = float(np.mean(errors))
    condition_number = float(np.linalg.cond(homography))
    if not np.isfinite(condition_number) or condition_number > 1e12:
        return _invalid_calibration(
            side,
            values,
            [*warnings, f"ill_conditioned_homography:{condition_number:.3e}"],
            point_count,
        )
    if point_count == 4:
        warnings.extend(
            [
                "minimum_four_point_fit_has_no_redundant_validation",
                "auxiliary_low_precision_calibration",
            ]
        )

    return CameraCalibration(
        side=side,
        calibration_id=values.get("calibration_id"),
        calibration_source=str(
            values.get("source", "manual_rough_fixed_view")
        ),
        image_size=configured_size,
        homography_image_to_court=homography.astype(np.float64),
        reprojection_error_px=mean_error,
        source_keypoint_count=point_count,
        valid=True,
        warnings=warnings,
    )


def load_camera_calibrations(
    values: dict,
    frame_sizes: dict[str, tuple[int, int]],
    layout: CourtLayout | None = None,
) -> dict[str, CameraCalibration]:
    """Build left/right calibration objects from the formal tracking config."""
    court_layout = layout or CourtLayout()
    cameras = values.get("cameras", {})
    return {
        side: _load_camera_calibration(
            side,
            dict(cameras.get(side, {})),
            court_layout,
            frame_sizes[side],
        )
        for side in ("left", "right")
    }
