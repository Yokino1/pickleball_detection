import unittest

import numpy as np

from src.court import (
    CourtLayout,
    CourtPanelRenderer,
    FixedCourtProjector,
    ProjectionResult,
    build_court_projection,
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

    def test_factory_uses_shared_court_net_deadband_for_events(self):
        court_values = calibration_config()
        court_values.update(
            {
                "enabled": True,
                "net_deadband_ft": 1.25,
            }
        )

        _, _, event_interpreter = build_court_projection(
            {"runtime": {"court_projection": court_values}},
            {"left": (200, 440), "right": (200, 440)},
        )

        self.assertIsNotNone(event_interpreter)
        self.assertAlmostEqual(event_interpreter.net_deadband_ft, 1.25)

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

    def test_projects_only_eligible_player_bbox_centers(self):
        centers = self.projector.project_eligible_player_box_centers(
            "left",
            [
                {
                    "eligible_player": True,
                    "bbox": [40.0, 80.0, 60.0, 120.0],
                },
                {
                    "eligible_player": False,
                    "bbox": [100.0, 100.0, 120.0, 140.0],
                },
                {"eligible_player": True, "bbox": [1.0, 2.0]},
            ],
        )

        self.assertEqual(len(centers), 1)
        self.assertTrue(np.allclose(centers[0], [5.0, 10.0]))

    def test_projects_player_bbox_center_beyond_court_boundary(self):
        centers = self.projector.project_eligible_player_box_centers(
            "left",
            [
                {
                    "eligible_player": True,
                    "bbox": [210.0, 80.0, 230.0, 120.0],
                }
            ],
        )

        self.assertEqual(len(centers), 1)
        self.assertTrue(np.allclose(centers[0], [22.0, 10.0]))
        self.assertFalse(self.layout.contains(centers[0]))

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

    def test_malformed_image_size_makes_projection_unavailable(self):
        config = calibration_config()
        config["cameras"]["left"]["image_size"] = []

        calibrations = load_camera_calibrations(
            config,
            {
                "left": (200, 440),
                "right": (200, 440),
            },
            self.layout,
        )

        self.assertFalse(calibrations["left"].valid)
        self.assertEqual(calibrations["left"].image_size, (0, 0))
        self.assertIn(
            "invalid_calibration_image_size",
            calibrations["left"].warnings,
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

    def test_score_text_uses_at_least_half_of_referee_panel_width(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=1000,
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

        panel = renderer.render(
            2160,
            projection,
            referee={
                "score": {"left": 12, "right": 11},
                "phase": "IN_RALLY",
                "server_side": "left",
                "service_owner_side": "left",
                "eligible_player_center_outside_court": True,
            },
        )

        score_region = panel[110:225, 24:-24]
        score_pixels = np.logical_and.reduce(
            (
                score_region[:, :, 0] < 80,
                score_region[:, :, 1] > 220,
                score_region[:, :, 2] < 80,
            )
        )
        _, columns = np.nonzero(score_pixels)
        self.assertGreater(columns.size, 0)
        self.assertGreaterEqual(
            int(columns.max() - columns.min() + 1),
            int(panel.shape[1] * 0.5),
        )

    def test_renderer_holds_second_bounce_feedback_and_merges_rejection(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=1000,
            court_event_hold_seconds=2.0,
        )
        second_bounce = renderer._visible_court_event(
            {
                "events": ["bounce_candidate", "second_bounce_candidate"],
                "contact_frame_index": 2537,
            },
            {
                "timestamp_s": 50.74,
                "phase": "SERVE_CONFIRMING",
                "events": [],
            },
        )

        self.assertIn(
            "second_bounce_candidate",
            second_bounce["court_events"],
        )
        merged = renderer._visible_court_event(
            {"events": []},
            {
                "timestamp_s": 50.96,
                "phase": "WAIT_SERVE",
                "events": [
                    "serve_rejected_player_center_inside_court"
                ],
            },
        )
        self.assertIn(
            "second_bounce_candidate",
            merged["court_events"],
        )
        self.assertIn(
            "serve_rejected_player_center_inside_court",
            merged["referee_events"],
        )

        held = renderer._visible_court_event(
            {"events": []},
            {"timestamp_s": 52.5, "phase": "WAIT_SERVE", "events": []},
        )
        expired = renderer._visible_court_event(
            {"events": []},
            {"timestamp_s": 53.0, "phase": "WAIT_SERVE", "events": []},
        )
        self.assertIsNotNone(held)
        self.assertIsNone(expired)

    def test_renderer_accumulates_each_score_reason_once(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=1000,
        )
        results = (
            {
                "result_id": "rally-1",
                "rally_id": 1,
                "point_awarded_to": "right",
                "fault_side": "left",
                "terminal_event": "return_net",
                "score_after": {"left": 0, "right": 1},
            },
            {
                "result_id": "rally-2",
                "rally_id": 2,
                "point_awarded_to": "left",
                "fault_side": "right",
                "terminal_event": "return_net",
                "score_after": {"left": 1, "right": 1},
            },
            {
                "result_id": "rally-3",
                "rally_id": 3,
                "point_awarded_to": "right",
                "fault_side": "left",
                "terminal_event": "out_of_bounds",
                "score_after": {"left": 1, "right": 2},
            },
        )
        for result in results:
            renderer._record_score_history(result)
        renderer._record_score_history(results[0])

        self.assertEqual(len(renderer._score_history), 3)
        self.assertEqual(
            [line[0] for line in renderer._score_history_lines()],
            [
                "1.  0:1   左侧下网",
                "2.  1:1   右侧下网",
                "3.  1:2   左侧出界",
            ],
        )

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

    def test_renderer_clips_extreme_finite_trail_coordinates(self):
        renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=360,
            trail_length=4,
        )

        def projection_at(x_ft, y_ft):
            return ProjectionResult(
                coordinate_system=self.layout.coordinate_system,
                coordinate_system_version=1,
                active_side="left",
                calibration_id="left_test",
                calibration_source="manual_test",
                image_xy=[100.0, 100.0],
                ball_court_xy=[x_ft, y_ft],
                projection_status="homography",
                projection_valid=True,
                homography_available=True,
                reprojection_error_px=0.0,
                track_status="observed",
                observed=True,
                inside_court=False,
            )

        renderer.render(720, projection_at(10.0, 22.0))
        panel = renderer.render(720, projection_at(1e20, 1e20))

        self.assertEqual(panel.shape, (720, 360, 3))

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

    def test_display_margin_changes_only_panel_scale(self):
        result = self.projector.project_track(
            "right",
            BallTrack(
                status="observed",
                center=[100.0, 330.0],
                source="detector",
            ),
        )
        original_court_xy = list(result.ball_court_xy)
        compact_renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=1000,
            outside_margin_ft=30.0,
        )
        enlarged_renderer = CourtPanelRenderer(
            self.layout,
            preferred_width=1000,
            outside_margin_ft=6.0,
        )
        compact_scale, _, _ = compact_renderer._transform(1000, 2160)
        enlarged_scale, _, _ = enlarged_renderer._transform(1000, 2160)

        enlarged_renderer.render(2160, result)

        self.assertGreater(enlarged_scale, compact_scale)
        self.assertEqual(result.ball_court_xy, original_court_xy)
        self.assertEqual(result.ball_court_xy, [10.0, 33.0])

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
