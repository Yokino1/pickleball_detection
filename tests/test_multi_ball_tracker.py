import unittest

from src.tracking.multi_ball_tracker import MultiBallTracker
from src.tracking.types import BallDetection


def detection(x: float, y: float, confidence: float = 0.9) -> BallDetection:
    return BallDetection(
        bbox=[x - 4, y - 4, x + 4, y + 4],
        center=[x, y],
        confidence=confidence,
        source="test",
    )


class MultiBallTrackerTest(unittest.TestCase):
    def test_predicts_through_short_detection_gap(self):
        tracker = MultiBallTracker(min_hits=1, max_predict_frames=3, max_missing_frames=5)
        first = tracker.update([detection(100, 100)], 640, 480)[0]
        second = tracker.update([detection(112, 100)], 640, 480)[0]
        predicted = tracker.update([], 640, 480)[0]

        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.track_id, predicted.track_id)
        self.assertEqual(predicted.status, "predicted")
        self.assertEqual(predicted.missing_frames, 1)
        self.assertGreater(predicted.center[0], first.center[0])

    def test_low_confidence_detection_only_recovers_existing_track(self):
        tracker = MultiBallTracker(high_conf=0.3, low_conf=0.08, min_hits=1)
        self.assertEqual(tracker.update([detection(50, 50, 0.1)], 640, 480), [])
        created = tracker.update([detection(50, 50, 0.8)], 640, 480)[0]
        recovered = tracker.update([detection(57, 50, 0.1)], 640, 480)[0]

        self.assertEqual(created.track_id, recovered.track_id)
        self.assertEqual(recovered.status, "observed")
        self.assertEqual(tracker.diagnostics["low_conf_recoveries"], 1)

    def test_tracks_multiple_balls_with_stable_ids(self):
        tracker = MultiBallTracker(min_hits=1)
        frame1 = tracker.update([detection(100, 120), detection(500, 250)], 640, 480)
        frame2 = tracker.update([detection(115, 120), detection(485, 250)], 640, 480)

        self.assertEqual([track.track_id for track in frame1], [1, 2])
        self.assertEqual([track.track_id for track in frame2], [1, 2])
        self.assertLess(frame2[0].center[0], frame2[1].center[0])

    def test_removes_stale_track_and_allocates_new_id(self):
        tracker = MultiBallTracker(
            min_hits=1, max_predict_frames=1, max_missing_frames=2
        )
        self.assertEqual(tracker.update([detection(20, 20)], 640, 480)[0].track_id, 1)
        tracker.update([], 640, 480)
        self.assertEqual(tracker.update([], 640, 480), [])
        tracker.update([], 640, 480)
        self.assertEqual(tracker.active_track_count, 0)
        self.assertEqual(tracker.update([detection(300, 200)], 640, 480)[0].track_id, 2)


if __name__ == "__main__":
    unittest.main()
