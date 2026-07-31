import unittest

import numpy as np

from src.court import (
    CourtLayout,
    CourtPanelRenderer,
    FixedCourtProjector,
    ProjectionResult,
    load_camera_calibrations,
)
from src.tracking.types import BallTrack, CourtInfo


def calibration_config():
    return {
        "cameras": {
            "left": {
                "calibration_id": "left_test",
                "source": "manual_test",
                "image_size": [200, 440],
                "keypoints": {
                    "kp0": [0.0, 0.0],
                    "kp2": [200.0, 0.0],
                    "kp6": [0.0, 220.0],
                    "kp7": [200.0, 220.0],
                },
            },
            "right": {
                "calibration_id": "right_test",
                "source": "manual_test",
                "image_size": [200, 440],
                "keypoints": {
                    "kp6": [0.0, 220.0],
                    "kp7": [200.0, 220.0],
                    "kp11": [0.0, 440.0],
                    "kp13": [200.0, 440.0],
                },
            },
        },
    }


class CourtProjectionTest(unittest.TestCase):
    def setUp(self):
        self.layout = CourtLayout()
        calibrations = load_camera_calibrations(
            calibration_config(),
            {
                "left": (200, 440),
                "right": (200, 440),
            },
            self.layout,
        )
        self.projector = FixedCourtProjector(calibrations, self.layout)

    def test_layout_uses_standard_full_court_coordinates(self):
        self.assertEqual(self.layout.width_ft, 20.0)
        self.assertEqual(self.layout.length_ft, 44.0)
        self.assertEqual(self.layout.net_y_ft, 22.0)
        self.assertEqual(self.layout.canonical_keypoints[13], (20.0, 44.0))

    def test_predicted_ball_outside_court_remains_projected(self):
        track = BallTrack(
            track_id=4,
            status="predicted",
            center=[250.0, 100.0],
            source="prediction",
        )

        result = self.projector.project_track("left", track)

        self.assertTrue(result.projection_valid)
        self.assertEqual(result.projection_status, "homography")
        self.assertTrue(result.predicted)
        self.assertFalse(result.observed)
        self.assertFalse(result.inside_court)
        self.assertAlmostEqual(result.ball_court_xy[0], 25.0, places=4)
        self.assertAlmostEqual(result.ball_court_xy[1], 10.0, places=4)
        self.assertIn("outside_court_bounds", result.projection_warnings)

    def test_observed_status_is_preserved(self):
        result = self.projector.project_track(
            "right",
            BallTrack(
                status="observed",
                center=[100.0, 330.0],
                source="detector",
            ),
        )

        self.assertTrue(result.observed)
        self.assertFalse(result.predicted)
        self.assertEqual(result.track_status, "observed")
        self.assertTrue(result.inside_court)
        self.assertTrue(np.allclose(result.ball_court_xy, [10.0, 33.0]))

    def test_image_size_mismatch_makes_projection_unavailable(self):
        calibrations = load_camera_calibrations(
            calibration_config(),
            {
                "left": (201, 440),
                "right": (200, 440),
            },
            self.layout,
        )
        projector = FixedCourtProjector(calibrations, self.layout)

        result = projector.project_track(
            "left",
            BallTrack(status="observed", center=[100.0, 100.0]),
        )

        self.assertFalse(result.projection_valid)
        self.assertFalse(result.homography_available)
        self.assertIn("homography_unavailable", result.projection_warnings)
        self.assertTrue(
            any(
                warning.startswith("calibration_image_size_mismatch")
                for warning in result.projection_warnings
            )
        )

    def test_renderer_is_blank_framework_without_camera_background(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=360,
            trail_length=4,
        )
        projection = ProjectionResult(
            coordinate_system=self.layout.coordinate_system,
            coordinate_system_version=1,
            active_side="left",
            calibration_id="left_test",
            calibration_source="manual_test",
            image_xy=[100.0, 100.0],
            ball_court_xy=[10.0, 10.0],
            projection_status="homography",
            projection_valid=True,
            homography_available=True,
            reprojection_error_px=0.0,
            track_status="observed",
            observed=True,
            inside_court=True,
        )

        panel = renderer.render(720, projection)

        self.assertEqual(panel.shape, (720, 360, 3))
        self.assertGreater(int(np.count_nonzero(panel[:, :, 1] == 255)), 0)
        flat = panel.reshape(-1, 3)
        blank_pixels = np.logical_or(
            np.all(flat == 24, axis=1),
            np.all(flat == 44, axis=1),
        )
        self.assertGreater(float(np.mean(blank_pixels)), 0.85)

    def test_renderer_marks_projection_beyond_viewport_at_edge(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=360,
            outside_margin_ft=4.0,
            trail_length=4,
        )
        projection = ProjectionResult(
            coordinate_system=self.layout.coordinate_system,
            coordinate_system_version=1,
            active_side="left",
            calibration_id="left_test",
            calibration_source="manual_test",
            image_xy=[100.0, 100.0],
            ball_court_xy=[1000.0, -1000.0],
            projection_status="homography",
            projection_valid=True,
            homography_available=True,
            reprojection_error_px=0.0,
            track_status="observed",
            observed=True,
            inside_court=False,
        )

        panel = renderer.render(720, projection)

        green = np.logical_and(
            panel[:, :, 1] > 220,
            panel[:, :, 2] < 80,
        )
        marker_rows, marker_columns = np.nonzero(green)
        self.assertGreater(marker_rows.size, 0)
        self.assertGreater(int(marker_rows.max()), panel.shape[0] - 40)
        self.assertLess(int(marker_columns.min()), 40)

    def test_renderer_rotates_court_length_to_horizontal_axis(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=800,
            outside_margin_ft=4.0,
        )
        scale, origin_x, origin_y = renderer._transform(800, 720)

        top_left = renderer._court_to_panel(
            (0.0, 0.0),
            scale,
            origin_x,
            origin_y,
        )
        bottom_right = renderer._court_to_panel(
            (20.0, 44.0),
            scale,
            origin_x,
            origin_y,
        )
        net_left = renderer._court_to_panel(
            (0.0, 22.0),
            scale,
            origin_x,
            origin_y,
        )
        net_right = renderer._court_to_panel(
            (20.0, 22.0),
            scale,
            origin_x,
            origin_y,
        )

        self.assertGreater(bottom_right[0], top_left[0])
        self.assertGreater(bottom_right[1], top_left[1])
        self.assertEqual(net_left[0], net_right[0])
        self.assertLess(net_left[1], net_right[1])

    def test_court_info_serializes_new_fields_without_removing_old_fields(self):
        record = CourtInfo(
            active_side="left",
            calibration_id="left_test",
            image_xy=[100.0, 120.0],
            projection_status="homography",
            projection_valid=True,
            homography_available=True,
            ball_court_xy=[10.0, 12.0],
            track_status="predicted",
            predicted=True,
            inside_court=True,
            event={
                "phase": "ground_contact_hold",
                "display_color": "yellow",
                "events": ["bounce_candidate"],
            },
        ).to_dict()

        self.assertEqual(record["layout"], "pickleball_14pt")
        self.assertEqual(record["ball_court_xy"], [10.0, 12.0])
        self.assertTrue(record["predicted"])
        self.assertEqual(record["event"]["display_color"], "yellow")
        self.assertIn("ball_zone", record)
        self.assertIn("visible_keypoints", record)


if __name__ == "__main__":
    unittest.main()
