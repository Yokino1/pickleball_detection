"""Project the final R9 global ball into one shared court coordinate system."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .calibration import CameraCalibration
from .layout import CourtLayout


@dataclass
class ProjectionResult:
    coordinate_system: str
    coordinate_system_version: int
    active_side: str | None
    calibration_id: str | None
    calibration_source: str | None
    image_xy: list[float] | None
    ball_court_xy: list[float] | None
    projection_status: str
    projection_valid: bool
    homography_available: bool
    reprojection_error_px: float | None
    projection_warnings: list[str] = field(default_factory=list)
    track_status: str | None = None
    observed: bool = False
    predicted: bool = False
    inside_court: bool | None = None
    event: dict | None = None

    def to_dict(self) -> dict:
        return {
            "coordinate_system": self.coordinate_system,
            "coordinate_system_version": self.coordinate_system_version,
            "active_side": self.active_side,
            "calibration_id": self.calibration_id,
            "calibration_source": self.calibration_source,
            "image_xy": self.image_xy,
            "ball_court_xy": self.ball_court_xy,
            "projection_status": self.projection_status,
            "projection_valid": self.projection_valid,
            "homography_available": self.homography_available,
            "reprojection_error_px": self.reprojection_error_px,
            "projection_warnings": list(self.projection_warnings),
            "track_status": self.track_status,
            "observed": self.observed,
            "predicted": self.predicted,
            "inside_court": self.inside_court,
            "event": dict(self.event) if self.event is not None else None,
        }


class FixedCourtProjector:
    """Read-only consumer of the coordinator's selected global ball."""

    def __init__(
        self,
        calibrations: dict[str, CameraCalibration],
        layout: CourtLayout | None = None,
    ):
        self.calibrations = dict(calibrations)
        self.layout = layout or CourtLayout()

    def is_side_available(self, side: str) -> bool:
        calibration = self.calibrations.get(side)
        return bool(calibration and calibration.homography_available)

    def calibration_for(self, side: str) -> CameraCalibration | None:
        return self.calibrations.get(side)

    def project_image_point(
        self,
        side: str | None,
        image_xy,
    ) -> tuple[float, float] | None:
        """Project one finite image point with the selected fixed homography."""
        calibration = self.calibrations.get(side) if side is not None else None
        if calibration is None or not calibration.homography_available:
            return None
        if not isinstance(image_xy, (list, tuple)) or len(image_xy) < 2:
            return None
        try:
            x_value = float(image_xy[0])
            y_value = float(image_xy[1])
        except (TypeError, ValueError, OverflowError):
            return None
        if not isfinite(x_value) or not isfinite(y_value):
            return None
        homography = calibration.homography_image_to_court
        assert homography is not None
        projected = homography @ np.asarray(
            [x_value, y_value, 1.0],
            dtype=np.float64,
        )
        denominator = float(projected[2])
        if not isfinite(denominator) or abs(denominator) < 1e-10:
            return None
        court_xy = (
            float(projected[0] / denominator),
            float(projected[1] / denominator),
        )
        if not all(isfinite(value) for value in court_xy):
            return None
        return court_xy

    def project_eligible_player_box_centers(
        self,
        side: str | None,
        players,
    ) -> list[tuple[float, float]]:
        """Project eligible-player bounding-box centers for serve evidence."""
        projected_centers: list[tuple[float, float]] = []
        for player in players or ():
            if not isinstance(player, dict):
                continue
            if not bool(player.get("eligible_player", False)):
                continue
            bbox = player.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                center = (
                    (float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            court_xy = self.project_image_point(side, center)
            if court_xy is not None:
                projected_centers.append(court_xy)
        return projected_centers

    def project_track(self, side: str | None, track) -> ProjectionResult:
        calibration = self.calibrations.get(side) if side is not None else None
        warnings = list(calibration.warnings) if calibration is not None else []
        result = ProjectionResult(
            coordinate_system=self.layout.coordinate_system,
            coordinate_system_version=self.layout.coordinate_system_version,
            active_side=side,
            calibration_id=(
                calibration.calibration_id if calibration is not None else None
            ),
            calibration_source=(
                calibration.calibration_source
                if calibration is not None
                else None
            ),
            image_xy=None,
            ball_court_xy=None,
            projection_status="unavailable",
            projection_valid=False,
            homography_available=bool(
                calibration and calibration.homography_available
            ),
            reprojection_error_px=(
                calibration.reprojection_error_px
                if calibration is not None
                else None
            ),
            projection_warnings=warnings,
            track_status=getattr(track, "status", None),
            observed=getattr(track, "status", None) == "observed",
            predicted=getattr(track, "status", None) == "predicted",
            inside_court=None,
        )
        if side not in ("left", "right"):
            result.projection_warnings.append("active_side_unavailable")
            return result
        if calibration is None or not calibration.homography_available:
            result.projection_warnings.append("homography_unavailable")
            return result
        center = getattr(track, "center", None)
        if track is None or center is None or len(center) < 2:
            result.projection_warnings.append("global_ball_unavailable")
            return result

        image_xy = [float(center[0]), float(center[1])]
        result.image_xy = image_xy
        homography = calibration.homography_image_to_court
        assert homography is not None
        projected = homography @ np.asarray(
            [image_xy[0], image_xy[1], 1.0],
            dtype=np.float64,
        )
        denominator = float(projected[2])
        if not isfinite(denominator) or abs(denominator) < 1e-10:
            result.projection_status = "invalid"
            result.projection_warnings.append("projection_denominator_invalid")
            return result
        court_xy = [
            float(projected[0] / denominator),
            float(projected[1] / denominator),
        ]
        if not all(isfinite(value) for value in court_xy):
            result.projection_status = "invalid"
            result.projection_warnings.append("projection_not_finite")
            return result

        inside_court = self.layout.contains((court_xy[0], court_xy[1]))
        result.ball_court_xy = court_xy
        result.projection_status = "homography"
        result.projection_valid = True
        result.inside_court = inside_court
        if not inside_court:
            result.projection_warnings.append("outside_court_bounds")
        return result
