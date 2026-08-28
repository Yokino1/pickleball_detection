import unittest

import numpy as np

from src.tracking.ball_detector import BallDetector
from src.tracking.ball_pipeline import BallTrackingPipeline, deduplicate_ball_detections
from src.tracking.multi_ball_tracker import MultiBallTracker
from src.tracking.person_detector import PersonDetector
from src.tracking.person_tracking import PersonBoxTracker, PlayerSelector
from src.tracking.types import BallDetection, PlayerDetection


class FakeDetector(BallDetector):
    def __init__(self):
        self.calls = 0

    def detect(self, frame, roi=None):
        self.calls += 1
        x = 100 + self.calls * 5
        return [BallDetection([x - 2, 98, x + 2, 102], [x, 100], 0.9, "fake")]


class FakePersonDetector(PersonDetector):
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [PlayerDetection([80, 40, 180, 220], confidence=0.9)]


class RoiOnlyDetector(BallDetector):
    def __init__(self):
        self.rois = []

    def detect(self, frame, roi=None):
        self.rois.append(roi)
        if roi is None:
            return []
        return [BallDetection([8, 8, 12, 12], [10, 10], 0.9, "roi_fake")]


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

    def test_passes_frame_timestamps_to_tracker(self):
        detector = FakeDetector()
        pipeline = BallTrackingPipeline(
            detector,
            MultiBallTracker(min_hits=1, max_prediction_ms=120, max_missing_ms=350),
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        pipeline.process_frame(frame, 0, timestamp_s=0.0)
        result = pipeline.process_frame(frame, 1, timestamp_s=0.05)

        self.assertAlmostEqual(result.diagnostics["tracker"]["frame_dt_ms"], 50.0)

    def test_handoff_search_roi_retries_detector_when_full_frame_misses(self):
        detector = RoiOnlyDetector()
        pipeline = BallTrackingPipeline(
            detector,
            MultiBallTracker(min_hits=1),
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        roi = [0.0, 0.0, 100.0, 240.0]

        result = pipeline.process_frame(frame, 0, search_roi=roi)

        self.assertEqual(detector.rois, [None, roi])
        self.assertTrue(result.diagnostics["roi_retry_used"])
        self.assertEqual(result.ball_detections[0].source, "roi_fake")

    def test_person_detector_runs_every_five_frames_and_boxes_persist(self):
        person_detector = FakePersonDetector()
        pipeline = BallTrackingPipeline(
            FakeDetector(),
            MultiBallTracker(min_hits=1),
            person_detector=person_detector,
            person_tracker=PersonBoxTracker(),
            player_selector=PlayerSelector(max_players=2),
            person_detector_interval=5,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        results = [
            pipeline.process_frame(frame, index, timestamp_s=index / 30)
            for index in range(6)
        ]

        self.assertEqual(person_detector.calls, 2)
        self.assertTrue(all(result.players for result in results))
        self.assertTrue(results[1].diagnostics["person_detector_skipped"])
        self.assertEqual(results[0].players[0].track_id, results[5].players[0].track_id)


if __name__ == "__main__":
    unittest.main()
