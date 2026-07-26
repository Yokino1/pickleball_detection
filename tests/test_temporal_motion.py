import unittest

import cv2
import numpy as np

from src.tracking.temporal_motion import TemporalMotionFilter
from src.tracking.types import BallDetection


def detection(x: float, y: float) -> BallDetection:
    return BallDetection(
        bbox=[x - 4, y - 4, x + 4, y + 4],
        center=[x, y],
        confidence=0.9,
        source="test",
    )


class TemporalMotionFilterTest(unittest.TestCase):
    def setUp(self):
        self.motion_filter = TemporalMotionFilter(
            analysis_width=320,
            difference_threshold=10,
            min_motion_fraction=0.01,
            min_region_radius_px=8,
            blur_kernel=3,
            dilation_iterations=2,
        )

    def test_first_frame_fails_open_then_static_candidate_is_rejected(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        candidate = detection(100, 90)

        self.assertEqual(self.motion_filter.filter(frame, [candidate]), [candidate])
        self.assertEqual(self.motion_filter.filter(frame, [candidate]), [])
        self.assertEqual(
            self.motion_filter.diagnostics["rejected_detections"],
            1,
        )

    def test_moving_candidate_has_local_motion_evidence(self):
        previous = np.zeros((180, 320, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        cv2.circle(previous, (80, 90), 4, (255, 255, 255), -1)
        cv2.circle(current, (100, 90), 4, (255, 255, 255), -1)
        candidate = detection(100, 90)

        self.motion_filter.filter(previous, [])
        accepted = self.motion_filter.filter(current, [candidate])

        self.assertEqual(accepted, [candidate])
        self.assertGreater(
            self.motion_filter.diagnostics["detection_scores"][0]["motion_fraction"],
            0.01,
        )

    def test_camera_translation_is_removed_before_differencing(self):
        previous = np.zeros((180, 320, 3), dtype=np.uint8)
        current = np.zeros_like(previous)
        cv2.circle(previous, (80, 90), 5, (255, 255, 255), -1)
        cv2.circle(current, (90, 90), 5, (255, 255, 255), -1)
        candidate = detection(90, 90)

        self.motion_filter.filter(previous, [])
        accepted = self.motion_filter.filter(
            current,
            [candidate],
            camera_motion=(10.0, 0.0),
        )

        self.assertEqual(accepted, [])


if __name__ == "__main__":
    unittest.main()
