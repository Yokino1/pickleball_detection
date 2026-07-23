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

    def test_static_track_sleeps_and_wakes_when_motion_resumes(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=3,
            max_stationary_frames=2,
        )
        self.assertEqual(tracker.update([detection(100, 100)], 1280, 720), [])
        self.assertEqual(tracker.update([detection(100, 100)], 1280, 720), [])
        moving = tracker.update([detection(110, 100)], 1280, 720)
        self.assertEqual(moving[0].track_id, 1)
        self.assertTrue(moving[0].motion_confirmed)

        tracker.update([detection(110, 100)], 1280, 720)
        tracker.update([detection(110, 100)], 1280, 720)
        self.assertEqual(tracker.update([detection(110, 100)], 1280, 720), [])
        self.assertEqual(tracker.diagnostics["stationary_suppressed_tracks"], 1)

        resumed = tracker.update([detection(120, 100)], 1280, 720)
        self.assertEqual(resumed[0].track_id, 1)
        self.assertEqual(resumed[0].stationary_frames, 0)

    def test_implausible_jump_cannot_move_an_existing_id(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=2,
            max_speed_px_per_frame=20,
            max_predict_frames=3,
        )
        self.assertEqual(tracker.update([detection(100, 100)], 1280, 720), [])
        observed = tracker.update([detection(110, 100)], 1280, 720)[0]
        after_jump = tracker.update([detection(170, 100)], 1280, 720)

        self.assertEqual(observed.track_id, 1)
        self.assertEqual([track.track_id for track in after_jump], [1])
        self.assertEqual(after_jump[0].status, "predicted")
        self.assertGreater(tracker.diagnostics["physical_gate_rejections"], 0)

    def test_active_ball_limit_hides_competing_stale_prediction(self):
        tracker = MultiBallTracker(min_hits=1, max_output_tracks=1, max_predict_frames=3)
        first = tracker.update([detection(100, 100)], 1280, 720)
        second = tracker.update([detection(500, 300)], 1280, 720)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].center, [500.0, 300.0])
        self.assertEqual(tracker.diagnostics["eligible_tracks"], 2)
        self.assertEqual(tracker.diagnostics["output_limited_tracks"], 1)

    def test_impact_recovery_keeps_id_after_fast_direction_change(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_speed_px_per_frame=120,
            impact_recovery_gate_px=260,
            impact_recovery_min_speed_px=10,
        )
        first = tracker.update([detection(800, 400)], 1280, 720)[0]
        tracker.update([detection(835, 430)], 1280, 720)
        tracker.update([], 1280, 720)
        recovered = tracker.update([detection(650, 285)], 1280, 720)

        self.assertEqual([track.track_id for track in recovered], [first.track_id])
        self.assertEqual(recovered[0].status, "observed")
        self.assertEqual(tracker.diagnostics["impact_recoveries"], 1)
        self.assertLess(recovered[0].velocity[0], 0)

    def test_fast_prediction_has_shorter_horizon(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_predict_frames=8,
            fast_prediction_speed_px=10,
            fast_max_predict_frames=2,
            prediction_min_confidence=0.1,
        )
        tracker.update([detection(100, 100)], 1280, 720)
        tracker.update([detection(140, 100)], 1280, 720)
        self.assertEqual(tracker.update([], 1280, 720)[0].missing_frames, 1)
        self.assertEqual(tracker.update([], 1280, 720)[0].missing_frames, 2)
        self.assertEqual(tracker.update([], 1280, 720), [])

    def test_camera_translation_does_not_activate_static_candidate(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=3,
            min_motion_speed_px=0.5,
        )
        self.assertEqual(tracker.update([detection(100, 100)], 1280, 720), [])
        self.assertEqual(
            tracker.update([detection(110, 100)], 1280, 720, camera_motion=(10, 0)),
            [],
        )
        self.assertEqual(
            tracker.update([detection(120, 100)], 1280, 720, camera_motion=(10, 0)),
            [],
        )

    def test_screen_fixed_overlay_does_not_activate_during_camera_motion(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=3,
            min_motion_speed_px=0.5,
        )
        self.assertEqual(tracker.update([detection(100, 100)], 1280, 720), [])
        self.assertEqual(
            tracker.update([detection(100, 100)], 1280, 720, camera_motion=(10, 0)),
            [],
        )
        self.assertEqual(
            tracker.update([detection(100, 100)], 1280, 720, camera_motion=(10, 0)),
            [],
        )

    def test_impact_recovery_handles_consecutive_frame_acceleration(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_speed_px_per_frame=120,
            impact_recovery_gate_px=260,
            impact_recovery_min_speed_px=10,
        )
        original = tracker.update([detection(700, 400)], 1280, 720)[0]
        tracker.update([detection(730, 420)], 1280, 720)
        recovered = tracker.update([detection(650, 360)], 1280, 720)

        self.assertEqual([track.track_id for track in recovered], [original.track_id])
        self.assertEqual(tracker.diagnostics["impact_recoveries"], 1)
        self.assertLess(recovered[0].velocity[0], 0)

    def test_impact_recovery_uses_average_speed_after_prediction_decay(self):
        tracker = MultiBallTracker(
            high_conf=0.25,
            low_conf=0.15,
            min_hits=1,
            max_predict_frames=8,
            max_missing_frames=20,
            base_gate_px=35.0,
            max_gate_px=80.0,
            gate_growth_px=10.0,
            require_motion_confirmation=False,
            impact_recovery_gate_px=260.0,
            impact_recovery_min_speed_px=15.0,
            prediction_velocity_decay=0.5,
        )
        tracker.update([detection(450.0, 150.0)], 1280, 720)
        tracker.update([detection(430.0, 150.0)], 1280, 720)
        track_id = tracker.update([detection(410.0, 150.0)], 1280, 720)[0].track_id
        for _ in range(6):
            tracker.update([], 1280, 720)

        outputs = tracker.update([detection(82.0, 150.0, confidence=0.35)], 1280, 720)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, track_id)
        self.assertEqual(outputs[0].status, "observed")

    def test_multiple_slow_rolling_balls_become_visible(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=6,
            min_motion_speed_px=0.5,
            max_stationary_frames=10,
        )
        outputs = []
        for offset in (0, 2, 4, 6, 8):
            outputs = tracker.update(
                [detection(100 + offset, 100), detection(400 - offset, 250)],
                1280,
                720,
            )

        self.assertEqual([track.track_id for track in outputs], [1, 2])
        self.assertTrue(all(track.motion_confirmed for track in outputs))


if __name__ == "__main__":
    unittest.main()
