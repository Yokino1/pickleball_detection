import unittest

import cv2
import numpy as np

from src.tracking.camera_motion import CameraMotionEstimator


class CameraMotionEstimatorTest(unittest.TestCase):
    def test_estimates_global_translation(self):
        first = np.zeros((240, 320, 3), dtype=np.uint8)
        for y in range(20, 221, 30):
            for x in range(20, 301, 30):
                cv2.rectangle(first, (x - 3, y - 3), (x + 3, y + 3), (255, 255, 255), -1)
        matrix = np.float32([[1, 0, 5], [0, 1, 3]])
        second = cv2.warpAffine(first, matrix, (320, 240))
        estimator = CameraMotionEstimator(
            analysis_width=320,
            min_inliers=10,
            max_shift_px=20,
        )

        estimator.update(first)
        motion = estimator.update(second)

        self.assertTrue(motion.applied)
        self.assertGreaterEqual(motion.inliers, 10)
        self.assertAlmostEqual(motion.dx, 5.0, delta=0.5)
        self.assertAlmostEqual(motion.dy, 3.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
