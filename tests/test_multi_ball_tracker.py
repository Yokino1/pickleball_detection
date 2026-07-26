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

    def test_frame_scale_override_preserves_crop_pixel_density(self):
        tracker = MultiBallTracker(
            min_hits=2,
            emit_tentative=False,
            require_motion_confirmation=True,
            motion_threshold_px=12,
            tentative_gate_px=150,
            max_speed_px_per_second=2400,
            reference_frame_width=1280,
            frame_scale_override=3.0,
            default_fps=50,
        )

        self.assertEqual(
            tracker.update(
                [detection(160, 700)],
                1920,
                1080,
                timestamp_s=0.0,
            ),
            [],
        )
        tracks = tracker.update(
            [detection(270, 700)],
            1920,
            1080,
            timestamp_s=0.02,
        )

        self.assertEqual([track.track_id for track in tracks], [1])
        self.assertEqual(tracks[0].status, "observed")
        self.assertEqual(tracker.diagnostics["physical_gate_rejections"], 0)
        self.assertEqual(tracker.diagnostics["frame_scale"], 3.0)
        self.assertTrue(tracker.diagnostics["frame_scale_overridden"])

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

    def test_contact_gate_allows_impact_recovery_only_near_player(self):
        def build():
            return MultiBallTracker(
                min_hits=1,
                max_speed_px_per_frame=120,
                impact_recovery_gate_px=260,
                impact_recovery_min_speed_px=10,
                require_contact_for_impact_recovery=True,
                contact_margin_ratio=0.15,
            )

        without_contact = build()
        original = without_contact.update([detection(800, 400)], 1280, 720)[0]
        without_contact.update([detection(835, 430)], 1280, 720)
        without_contact.update([], 1280, 720)
        rejected = without_contact.update([detection(650, 285)], 1280, 720)

        self.assertFalse(
            any(
                track.track_id == original.track_id and track.status == "observed"
                for track in rejected
            )
        )
        self.assertGreater(without_contact.diagnostics["contact_gate_rejections"], 0)

        with_contact = build()
        original = with_contact.update([detection(800, 400)], 1280, 720)[0]
        with_contact.update([detection(835, 430)], 1280, 720)
        with_contact.update([], 1280, 720)
        recovered = with_contact.update(
            [detection(650, 285)],
            1280,
            720,
            contact_zones=[[780, 330, 900, 600]],
        )

        self.assertTrue(
            any(
                track.track_id == original.track_id and track.status == "observed"
                for track in recovered
            )
        )
        self.assertEqual(with_contact.diagnostics["impact_recoveries"], 1)

    def test_contact_gate_rejects_distant_segment_crossing_player_box(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_speed_px_per_second=10000,
            impact_recovery_gate_px=500,
            impact_recovery_min_speed_px_per_second=10,
            require_contact_for_impact_recovery=True,
            contact_margin_ratio=0.15,
        )
        original = tracker.update([detection(600, 400)], 1280, 720)[0]
        tracker.update([detection(650, 400)], 1280, 720)
        tracker.update([], 1280, 720)
        outputs = tracker.update(
            [detection(1000, 400)],
            1280,
            720,
            contact_zones=[[850, 330, 900, 600]],
        )

        self.assertFalse(
            any(
                item.track_id == original.track_id
                and item.status == "observed"
                for item in outputs
            )
        )
        self.assertGreater(
            tracker.diagnostics["contact_gate_rejections"],
            0,
        )

    def test_impact_recovery_expires_after_configured_time(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_prediction_ms=60,
            max_missing_ms=350,
            max_speed_px_per_second=5000,
            impact_recovery_gate_px=500,
            impact_recovery_min_speed_px_per_second=10,
            impact_recovery_max_missing_ms=120,
        )
        original = tracker.update(
            [detection(100, 400)],
            1280,
            720,
            timestamp_s=0.0,
        )[0]
        tracker.update(
            [detection(110, 400)],
            1280,
            720,
            timestamp_s=0.02,
        )
        for timestamp in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16):
            tracker.update([], 1280, 720, timestamp_s=timestamp)
        outputs = tracker.update(
            [detection(500, 400)],
            1280,
            720,
            timestamp_s=0.18,
        )

        self.assertFalse(
            any(
                item.track_id == original.track_id
                and item.status == "observed"
                for item in outputs
            )
        )
        self.assertEqual(tracker.diagnostics["impact_recoveries"], 0)

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

    def test_time_based_prediction_has_same_duration_across_frame_rates(self):
        def visible_prediction_times(fps: float) -> list[float]:
            tracker = MultiBallTracker(
                min_hits=1,
                max_prediction_ms=120,
                max_missing_ms=350,
                fast_prediction_speed_px_per_second=0,
            )
            tracker.update([detection(100, 100)], 1280, 720, timestamp_s=0.0)
            tracker.update([detection(110, 100)], 1280, 720, timestamp_s=1.0 / fps)
            visible = []
            frame = 2
            while frame < 20:
                timestamp = frame / fps
                outputs = tracker.update([], 1280, 720, timestamp_s=timestamp)
                if not outputs:
                    break
                visible.append(outputs[0].missing_time_ms)
                frame += 1
            return visible

        at_20_fps = visible_prediction_times(20.0)
        at_60_fps = visible_prediction_times(60.0)

        self.assertEqual(len(at_20_fps), 2)
        self.assertAlmostEqual(at_20_fps[0], 50.0)
        self.assertAlmostEqual(at_20_fps[1], 100.0)
        self.assertAlmostEqual(at_60_fps[-1], 116.667, places=2)
        self.assertTrue(all(value <= 120.0 for value in at_60_fps))

    def test_speed_gate_is_px_per_second_and_resolution_scaled(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_speed_px_per_second=1000,
            max_prediction_ms=120,
            max_missing_ms=350,
        )
        tracker.update([detection(100, 100)], 640, 360, timestamp_s=0.0)
        accepted = tracker.update(
            [detection(145, 100)],
            640,
            360,
            timestamp_s=0.1,
        )
        rejected = tracker.update(
            [detection(220, 100)],
            640,
            360,
            timestamp_s=0.2,
        )

        self.assertEqual(accepted[0].status, "observed")
        self.assertEqual(rejected[0].status, "predicted")
        self.assertGreater(tracker.diagnostics["physical_gate_rejections"], 0)

    def test_fast_prediction_uses_milliseconds(self):
        tracker = MultiBallTracker(
            min_hits=1,
            max_prediction_ms=150,
            max_missing_ms=350,
            fast_prediction_speed_px_per_second=500,
            fast_max_prediction_ms=60,
        )
        tracker.update([detection(100, 100)], 1280, 720, timestamp_s=0.0)
        tracker.update([detection(180, 100)], 1280, 720, timestamp_s=0.05)

        first_miss = tracker.update([], 1280, 720, timestamp_s=0.10)
        second_miss = tracker.update([], 1280, 720, timestamp_s=0.15)

        self.assertEqual(first_miss[0].status, "predicted")
        self.assertAlmostEqual(first_miss[0].missing_time_ms, 50.0)
        self.assertEqual(second_miss, [])

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

    def test_non_impact_direction_jump_is_rejected(self):
        tracker = MultiBallTracker(
            min_hits=1,
            require_motion_confirmation=True,
            motion_threshold_px=2,
            min_motion_speed_px_per_second=20,
            max_speed_px_per_second=3000,
            max_flight_direction_change_deg=60,
            direction_gate_min_speed_px_per_second=100,
            direction_gate_min_hits=3,
            impact_recovery_min_missing_ms=20,
        )
        tracker.update([detection(100, 100)], 1280, 720, timestamp_s=0.0)
        tracker.update([detection(120, 100)], 1280, 720, timestamp_s=1 / 30)
        tracker.update([detection(140, 100)], 1280, 720, timestamp_s=2 / 30)

        result = tracker.update(
            [detection(140, 75)],
            1280,
            720,
            timestamp_s=3 / 30,
        )

        self.assertEqual(result[0].status, "predicted")
        self.assertEqual(tracker.diagnostics["impact_recoveries"], 0)
        self.assertEqual(tracker.diagnostics["direction_gate_rejections"], 1)

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

    def test_constant_acceleration_model_emits_bounded_acceleration(self):
        tracker = MultiBallTracker(
            min_hits=1,
            motion_model="constant_acceleration",
            constant_acceleration_min_observations=3,
            max_acceleration_px_per_second2=8000,
            acceleration_process_noise=3000,
            max_speed_px_per_second=5000,
        )
        points = [(100, 100), (110, 101), (120, 104), (130, 109)]
        output = None
        for index, (x, y) in enumerate(points):
            output = tracker.update(
                [detection(x, y)],
                1280,
                720,
                timestamp_s=index * 0.05,
            )[0]

        self.assertIsNotNone(output)
        self.assertEqual(output.motion_model, "constant_acceleration")
        self.assertGreater(output.acceleration[1], 0.0)
        self.assertLessEqual(
            (output.acceleration[0] ** 2 + output.acceleration[1] ** 2) ** 0.5,
            8000.0,
        )

    def test_nis_gate_rejects_unexpected_non_impact_measurement(self):
        tracker = MultiBallTracker(
            min_hits=1,
            process_noise=10,
            measurement_noise=2,
            max_speed_px_per_second=5000,
            use_nis_gate=True,
            nis_gate_threshold=5.0,
            nis_gate_min_hits=3,
            impact_recovery_min_missing_ms=20,
        )
        tracker.update([detection(100, 100)], 1280, 720, timestamp_s=0.0)
        tracker.update([detection(110, 100)], 1280, 720, timestamp_s=0.05)
        tracker.update([detection(120, 100)], 1280, 720, timestamp_s=0.10)

        output = tracker.update(
            [detection(160, 125)],
            1280,
            720,
            timestamp_s=0.15,
        )

        self.assertEqual(output[0].status, "predicted")
        self.assertEqual(tracker.diagnostics["nis_gate_rejections"], 1)

    def test_continuous_prediction_horizon_shortens_with_speed(self):
        tracker = MultiBallTracker(
            max_prediction_ms=120,
            max_missing_ms=350,
            fast_prediction_speed_px_per_second=800,
            fast_max_prediction_ms=60,
            continuous_prediction_horizon=True,
        )

        self.assertAlmostEqual(tracker._prediction_limit_s(0), 0.12)
        self.assertAlmostEqual(tracker._prediction_limit_s(400), 0.09)
        self.assertAlmostEqual(tracker._prediction_limit_s(800), 0.06)


if __name__ == "__main__":
    unittest.main()
