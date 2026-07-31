import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "ground_detection"
    / "pickleball_court_detector_handoff.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pickleball_court_detector_handoff",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
COURT_DETECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COURT_DETECTOR)


class FullCourtCalibrationCompletionTest(unittest.TestCase):
    def setUp(self):
        self.court_to_image = np.asarray(
            [
                [18.0, 2.5, 120.0],
                [0.8, 9.0, 80.0],
                [0.0015, 0.0020, 1.0],
            ],
            dtype=np.float64,
        )

    def image_point(self, point_id):
        court_xy = COURT_DETECTOR.CANONICAL_KEYPOINTS_FT[point_id]
        result = COURT_DETECTOR.transform_point(
            court_xy,
            self.court_to_image,
        )
        self.assertIsNotNone(result)
        return result

    def test_video_input_reads_only_frame_zero(self):
        expected = np.zeros((24, 32, 3), dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [50.0, 3030.0]
        capture.read.return_value = (True, expected)

        with patch.object(
            COURT_DETECTOR.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            frame, info = COURT_DETECTOR.read_calibration_frame(
                Path("fixed_camera_left.mp4"),
            )

        self.assertIs(frame, expected)
        capture.read.assert_called_once_with()
        capture.release.assert_called_once_with()
        self.assertEqual(info["input_kind"], "video")
        self.assertEqual(info["input_frame_index"], 0)
        self.assertEqual(info["source_video_fps"], 50.0)
        self.assertEqual(info["source_video_frame_count"], 3030)

    def test_rough_manual_calibration_is_explicitly_marked(self):
        completion = COURT_DETECTOR.complete_from_rough_manual_keypoints(
            {"half_side": "left"},
            (720, 1280),
            {
                "calibration_id": "left_rough_test",
                "image_size": [1280, 720],
                "keypoints": {
                    "kp0": [100.0, 200.0],
                    "kp2": [100.0, 600.0],
                    "kp6": [1100.0, 250.0],
                    "kp7": [1100.0, 550.0],
                },
            },
        )

        self.assertTrue(completion["valid"])
        self.assertEqual(
            completion["calibration_method"],
            "manual_rough_fixed_view",
        )
        self.assertEqual(completion["calibration_id"], "left_rough_test")
        self.assertIn(
            "auxiliary_low_precision_calibration",
            completion["warnings"],
        )

    def test_topdown_projection_has_standard_full_court_canvas(self):
        completion = COURT_DETECTOR.complete_from_rough_manual_keypoints(
            {"half_side": "left"},
            (720, 1280),
            {
                "calibration_id": "left_topdown_test",
                "image_size": [1280, 720],
                "keypoints": {
                    "kp0": [100.0, 100.0],
                    "kp2": [100.0, 620.0],
                    "kp6": [1180.0, 150.0],
                    "kp7": [1180.0, 570.0],
                },
            },
        )
        frame = np.full((720, 1280, 3), 160, dtype=np.uint8)

        topdown, metadata = COURT_DETECTOR.render_topdown_projection(
            frame,
            completion,
            "left",
            pixels_per_foot=10.0,
            margin_px=20,
        )

        self.assertEqual(topdown.shape, (480, 240, 3))
        self.assertEqual(metadata["status"], "available")
        self.assertEqual(metadata["court_bounds_ft"], [0.0, 0.0, 20.0, 44.0])
        self.assertIsNotNone(metadata["image_to_topdown"])

    def test_topdown_stitch_uses_left_above_net_and_right_below(self):
        left = np.zeros((480, 240, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        left[:] = (255, 0, 0)
        right[:] = (0, 255, 0)

        stitched = COURT_DETECTOR.stitch_topdown_projections(
            left,
            right,
            pixels_per_foot=10.0,
            margin_px=20,
        )

        self.assertGreater(int(stitched[100, 100, 0]), 0)
        self.assertGreater(int(stitched[360, 100, 1]), 0)

    def test_left_half_points_complete_all_fourteen_keypoints(self):
        source_ids = [f"kp{index}" for index in range(8)]
        reconstruction = {
            "half_side": "left",
            "target_keypoints": [
                {
                    "id": point_id,
                    "status": "observed",
                    "source": "synthetic",
                    "point": self.image_point(point_id),
                }
                for point_id in source_ids
            ],
        }

        completion = COURT_DETECTOR.complete_full_court_calibration(
            reconstruction,
            (720, 1280),
        )

        self.assertEqual(completion["status"], "homography")
        self.assertTrue(completion["valid"])
        self.assertEqual(completion["source_keypoint_ids"], source_ids)
        self.assertEqual(len(completion["completed_keypoints_14"]), 14)
        self.assertLess(
            completion["quality"]["mean_reprojection_error_px"],
            1e-3,
        )

        completed = {
            point["id"]: point
            for point in completion["completed_keypoints_14"]
        }
        for point_id in COURT_DETECTOR.CANONICAL_KEYPOINTS_FT:
            expected = self.image_point(point_id)
            actual = completed[point_id]["homography_point"]
            self.assertTrue(np.allclose(actual, expected, atol=1e-3))
        self.assertEqual(
            completed["kp13"]["status"],
            "homography_completed",
        )
        self.assertFalse(completed["kp13"]["used_for_homography"])

    def test_image_to_court_matrix_round_trips_a_point(self):
        reconstruction = {
            "half_side": "right",
            "target_keypoints": [
                {
                    "id": f"kp{index}",
                    "status": "inferred",
                    "source": "synthetic",
                    "point": self.image_point(f"kp{index}"),
                }
                for index in range(6, 14)
            ],
        }
        completion = COURT_DETECTOR.complete_full_court_calibration(
            reconstruction,
            (720, 1280),
        )
        image_xy = self.image_point("kp4")
        court_xy = COURT_DETECTOR.transform_point(
            image_xy,
            completion["homography_image_to_court"],
        )
        self.assertIsNotNone(court_xy)
        self.assertTrue(
            np.allclose(
                court_xy,
                COURT_DETECTOR.CANONICAL_KEYPOINTS_FT["kp4"],
                atol=1e-4,
            )
        )

    def test_too_few_points_return_unavailable_without_coordinates(self):
        reconstruction = {
            "half_side": "left",
            "target_keypoints": [
                {
                    "id": point_id,
                    "status": "observed",
                    "point": self.image_point(point_id),
                }
                for point_id in ("kp0", "kp1", "kp3")
            ],
        }
        completion = COURT_DETECTOR.complete_full_court_calibration(
            reconstruction,
            (720, 1280),
        )
        self.assertEqual(completion["status"], "unavailable")
        self.assertFalse(completion["valid"])
        self.assertEqual(completion["completed_keypoints_14"], [])
        self.assertIn(
            "too_few_semantic_points:3",
            completion["warnings"],
        )

    def test_collinear_semantic_points_are_invalid(self):
        source_ids = ("kp0", "kp3", "kp6", "kp8", "kp11")
        reconstruction = {
            "half_side": "left",
            "target_keypoints": [
                {
                    "id": point_id,
                    "status": "inferred",
                    "point": self.image_point(point_id),
                }
                for point_id in source_ids
            ],
        }
        completion = COURT_DETECTOR.complete_full_court_calibration(
            reconstruction,
            (720, 1280),
        )
        self.assertEqual(completion["status"], "invalid")
        self.assertFalse(completion["valid"])
        self.assertEqual(completion["completed_keypoints_14"], [])
        self.assertIn("degenerate_semantic_points", completion["warnings"])

    def test_full_court_visualization_accepts_completed_points(self):
        source_ids = [f"kp{index}" for index in range(8)]
        completion = COURT_DETECTOR.complete_full_court_calibration(
            {
                "half_side": "left",
                "target_keypoints": [
                    {
                        "id": point_id,
                        "status": "observed",
                        "point": self.image_point(point_id),
                    }
                    for point_id in source_ids
                ],
            },
            (720, 1280),
        )
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = COURT_DETECTOR.draw_full_court_calibration(
            image,
            completion,
        )
        self.assertEqual(rendered.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(rendered)), 0)


if __name__ == "__main__":
    unittest.main()
