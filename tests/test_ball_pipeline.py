import unittest

import numpy as np

from src.tracking.ball_detector import BallDetector
from src.tracking.ball_pipeline import BallTrackingPipeline, deduplicate_ball_detections
from src.tracking.multi_ball_tracker import MultiBallTracker
from src.tracking.types import BallDetection


class FakeDetector(BallDetector):
    def __init__(self):
        self.calls = 0

    def detect(self, frame, roi=None):
        self.calls += 1
        x = 100 + self.calls * 5
        return [BallDetection([x - 2, 98, x + 2, 102], [x, 100], 0.9, "fake")]


class BallTrackingPipelineTest(unittest.TestCase):
    def test_deduplicates_nested_boxes_for_the_same_ball(self):
        strongest = BallDetection([90, 90, 110, 110], [100, 100], 0.9, "fake")
        nested = BallDetection([94, 94, 106, 106], [100, 100], 0.7, "fake")
        separate = BallDetection([190, 90, 210, 110], [200, 100], 0.8, "fake")

        result = deduplicate_ball_detections([nested, separate, strongest])

        self.assertEqual(result, [strongest, separate])

    def test_detector_interval_uses_prediction_between_detector_frames(self):
        detector = FakeDetector()
        pipeline = BallTrackingPipeline(
            detector,
            MultiBallTracker(min_hits=1, max_predict_frames=3),
            detector_interval=2,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        first = pipeline.process_frame(frame, 0)
        second = pipeline.process_frame(frame, 1)
        third = pipeline.process_frame(frame, 2)

        self.assertEqual(detector.calls, 2)
        self.assertEqual(first.ball_tracks[0].status, "observed")
        self.assertEqual(second.ball_tracks[0].status, "predicted")
        self.assertEqual(third.ball_tracks[0].status, "observed")
        self.assertTrue(second.diagnostics["detector_skipped"])
        serialized = second.to_dict()
        self.assertEqual(serialized["ball_tracks"][0]["track_id"], 1)
        self.assertEqual(serialized["ball_tracks"][0]["status"], "predicted")


if __name__ == "__main__":
    unittest.main()
