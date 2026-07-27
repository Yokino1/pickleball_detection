import unittest

from src.tracking.dual_camera.coordinator import (
    CrossCameraHandoffAdvisor,
    CrossHalfBallCoordinator,
    GlobalBallSelection,
)
from src.tracking.types import BallTrack


def track(
    track_id: int,
    *,
    status: str = "observed",
    confidence: float = 0.8,
    missing_time_ms: float = 0.0,
    center: tuple[float, float] = (100.0, 120.0),
    source: str = "detector",
    observation_source: str | None = None,
) -> BallTrack:
    return BallTrack(
        track_id=track_id,
        status=status,
        center=[float(center[0]), float(center[1])],
        bbox=[
            float(center[0]) - 4.0,
            float(center[1]) - 4.0,
            float(center[0]) + 4.0,
            float(center[1]) + 4.0,
        ],
        confidence=confidence,
        missing_time_ms=missing_time_ms,
        source=source,
        observation_source=observation_source,
        hits=5,
        confirmed=True,
        motion_confirmed=True,
    )


def handoff(
    source_side: str,
    target_side: str,
    *,
    expires_at_s: float = 1.0,
    target_roi: list[float] | None = None,
) -> dict:
    return {
        "active": True,
        "source_side": source_side,
        "target_side": target_side,
        "expires_at_s": expires_at_s,
        "target_roi": target_roi or [0.0, 0.0, 300.0, 600.0],
    }


class CrossHalfBallCoordinatorTest(unittest.TestCase):
    def test_selects_only_one_global_ball(self):
        coordinator = CrossHalfBallCoordinator()
        result = coordinator.update([track(7, confidence=0.9)], [track(3, confidence=0.7)])

        self.assertEqual(result.active_side, "left")
        self.assertEqual(result.track.track_id, 1)
        self.assertEqual(result.local_track_id, 7)

    def test_active_observed_side_has_hysteresis(self):
        coordinator = CrossHalfBallCoordinator()
        coordinator.update([track(7, confidence=0.9)], [])
        result = coordinator.update(
            [track(7, confidence=0.5)],
            [track(3, confidence=0.99)],
        )

        self.assertEqual(result.active_side, "left")
        self.assertFalse(result.switched_side)

    def test_switches_side_after_active_track_is_missing(self):
        coordinator = CrossHalfBallCoordinator(switch_min_missing_ms=20.0)
        coordinator.update([track(7)], [])
        result = coordinator.update(
            [track(7, status="predicted", missing_time_ms=40.0)],
            [track(3, confidence=0.75)],
        )

        self.assertEqual(result.active_side, "right")
        self.assertEqual(result.track.track_id, 1)
        self.assertEqual(result.local_track_id, 3)
        self.assertTrue(result.switched_side)
        self.assertEqual(result.previous_side, "left")
        self.assertEqual(coordinator.diagnostics()["handoff_count"], 1)

    def test_reacquires_new_local_id_on_active_side_before_switching(self):
        coordinator = CrossHalfBallCoordinator()
        coordinator.update([track(7)], [])
        result = coordinator.update(
            [track(8, confidence=0.7)],
            [track(3, confidence=0.99)],
        )

        self.assertEqual(result.active_side, "left")
        self.assertEqual(result.local_track_id, 8)
        self.assertFalse(result.switched_side)

    def test_rejects_impossible_same_side_global_jump(self):
        coordinator = CrossHalfBallCoordinator(
            max_continuity_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 3.0},
        )
        widths = {"left": 1920, "right": 1920}
        coordinator.update(
            [track(7, center=(100.0, 500.0))],
            [],
            timestamp_s=0.0,
            frame_widths=widths,
        )

        rejected = coordinator.update(
            [track(8, center=(600.0, 500.0))],
            [],
            timestamp_s=0.02,
            frame_widths=widths,
        )
        recovered = coordinator.update(
            [track(7, center=(130.0, 500.0))],
            [],
            timestamp_s=0.04,
            frame_widths=widths,
        )

        self.assertIsNone(rejected.track)
        self.assertEqual(rejected.state, "temporarily_lost")
        self.assertEqual(rejected.local_track_id, 7)
        self.assertEqual(recovered.local_track_id, 7)
        self.assertEqual(recovered.track.status, "observed")
        self.assertEqual(
            coordinator.diagnostics()["continuity_gate_rejections"],
            1,
        )

    def test_rejects_impossible_correction_with_same_local_id(self):
        coordinator = CrossHalfBallCoordinator(
            max_continuity_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 3.0},
        )
        widths = {"left": 1920, "right": 1920}
        coordinator.update(
            [track(7, status="observed", center=(300.0, 900.0))],
            [],
            timestamp_s=0.0,
            frame_widths=widths,
        )

        rejected = coordinator.update(
            [track(7, status="observed", center=(800.0, 1000.0))],
            [],
            timestamp_s=0.02,
            frame_widths=widths,
        )

        self.assertIsNone(rejected.track)
        self.assertEqual(rejected.state, "temporarily_lost")
        self.assertEqual(
            coordinator.diagnostics()["continuity_gate_rejections"],
            1,
        )

    def test_does_not_mutate_local_track_id(self):
        coordinator = CrossHalfBallCoordinator()
        local = track(12)
        result = coordinator.update([local], [])

        self.assertEqual(local.track_id, 12)
        self.assertEqual(result.track.track_id, 1)

    def test_net_bound_ball_activates_receiving_camera_roi(self):
        advisor = CrossCameraHandoffAdvisor(
            net_margin_ratio=0.2,
            receiving_band_ratio=0.3,
            min_toward_net_speed_px_per_second=200,
            alert_duration_ms=100,
        )
        selected_track = track(1)
        selected_track.center = [920.0, 300.0]
        selected_track.velocity = [300.0, 0.0]
        selection = GlobalBallSelection(
            global_track_id=1,
            active_side="left",
            state="left_active",
            track=selected_track,
            local_track_id=7,
        )

        advisor.update(
            selection,
            timestamp_s=1.0,
            frame_widths={"left": 1000, "right": 1000},
        )
        roi = advisor.search_roi(
            "right",
            timestamp_s=1.05,
            frame_width=1000,
            frame_height=600,
        )

        self.assertEqual(roi, [0.0, 0.0, 300.0, 600.0])
        self.assertIsNone(
            advisor.search_roi(
                "left",
                timestamp_s=1.05,
                frame_width=1000,
                frame_height=600,
            )
        )
        self.assertIsNone(
            advisor.search_roi(
                "right",
                timestamp_s=1.2,
                frame_width=1000,
                frame_height=600,
            )
        )

    def test_handoff_uses_configured_stream_scale(self):
        advisor = CrossCameraHandoffAdvisor(
            net_margin_ratio=0.2,
            min_toward_net_speed_px_per_second=200,
            frame_scale_overrides={"left": 3.0},
        )
        selected_track = track(1)
        selected_track.center = [920.0, 300.0]
        selected_track.velocity = [300.0, 0.0]
        selection = GlobalBallSelection(
            global_track_id=1,
            active_side="left",
            state="left_active",
            track=selected_track,
            local_track_id=7,
        )

        advisor.update(
            selection,
            timestamp_s=1.0,
            frame_widths={"left": 1000, "right": 1000},
        )

        self.assertIsNone(
            advisor.search_roi(
                "right",
                timestamp_s=1.05,
                frame_width=1000,
                frame_height=600,
            )
        )

    def test_strict_handoff_rejects_receiver_without_armed_window(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            receiver_confirmation_hits=2,
        )
        coordinator.update(
            [track(7)],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [track(7, status="predicted", missing_time_ms=40.0)],
            [track(3, center=(100.0, 300.0))],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "left")
        self.assertFalse(result.switched_side)
        self.assertEqual(result.local_track_id, 7)
        self.assertEqual(coordinator.diagnostics()["handoff_rejections"], 1)

    def test_primary_observation_preempts_prediction_without_handoff(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            observation_first=True,
        )
        coordinator.update(
            [track(7, observation_source="yolo")],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [
                track(
                    7,
                    status="predicted",
                    missing_time_ms=20.0,
                    source="prediction",
                )
            ],
            [
                track(
                    3,
                    center=(700.0, 300.0),
                    observation_source="yolo",
                )
            ],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "right")
        self.assertEqual(result.track.status, "observed")
        self.assertEqual(result.track.source, "detector")
        self.assertEqual(result.track.observation_source, "yolo")
        self.assertTrue(result.switched_side)
        self.assertEqual(
            coordinator.diagnostics()["observation_preemptions"],
            1,
        )

    def test_same_side_primary_observation_preempts_old_prediction(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            observation_first=True,
        )
        coordinator.update(
            [track(7, observation_source="yolo")],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [
                track(
                    7,
                    status="predicted",
                    missing_time_ms=20.0,
                    source="prediction",
                ),
                track(
                    8,
                    center=(112.0, 118.0),
                    observation_source="yolo",
                ),
            ],
            [],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "left")
        self.assertEqual(result.local_track_id, 8)
        self.assertEqual(result.track.status, "observed")
        self.assertFalse(result.switched_side)
        self.assertEqual(
            coordinator.diagnostics()[
                "same_side_observation_preemptions"
            ],
            1,
        )

    def test_primary_observation_preempts_missing_active_side(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            observation_first=True,
        )
        coordinator.update(
            [track(7, observation_source="yolo")],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [],
            [
                track(
                    3,
                    center=(700.0, 300.0),
                    observation_source="onnxruntime",
                )
            ],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "right")
        self.assertEqual(result.track.status, "observed")
        self.assertEqual(result.track.source, "detector")
        self.assertEqual(result.track.observation_source, "onnxruntime")
        self.assertTrue(result.switched_side)

    def test_primary_observation_is_not_vetoed_by_duplicate_global_speed_gate(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            observation_first=True,
            max_continuity_speed_px_per_second=100.0,
        )
        coordinator.update(
            [track(7, center=(100.0, 300.0), observation_source="yolo")],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [track(7, center=(900.0, 300.0), observation_source="yolo")],
            [],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "left")
        self.assertEqual(result.track.status, "observed")
        self.assertEqual(
            coordinator.diagnostics()["continuity_gate_rejections"],
            0,
        )

    def test_fast_motion_observation_cannot_bypass_handoff(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            observation_first=True,
        )
        coordinator.update(
            [track(7, observation_source="yolo")],
            [],
            timestamp_s=0.0,
        )

        result = coordinator.update(
            [
                track(
                    7,
                    status="predicted",
                    missing_time_ms=40.0,
                    source="prediction",
                )
            ],
            [
                track(
                    3,
                    center=(100.0, 300.0),
                    observation_source="fast_motion",
                )
            ],
            timestamp_s=0.02,
        )

        self.assertEqual(result.active_side, "left")
        self.assertEqual(result.track.status, "predicted")
        self.assertFalse(result.switched_side)
        self.assertEqual(
            coordinator.diagnostics()["observation_preemptions"],
            0,
        )

    def test_strict_handoff_requires_consecutive_receiver_confirmation(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            receiver_confirmation_hits=2,
        )
        widths = {"left": 1000, "right": 1000}
        coordinator.update(
            [track(7)],
            [],
            timestamp_s=0.0,
            frame_widths=widths,
        )

        confirming = coordinator.update(
            [track(7, status="predicted", missing_time_ms=40.0)],
            [track(3, center=(100.0, 300.0))],
            timestamp_s=0.02,
            frame_widths=widths,
            handoff=handoff("left", "right"),
        )
        switched = coordinator.update(
            [],
            [track(3, center=(112.0, 300.0))],
            timestamp_s=0.04,
            frame_widths=widths,
            handoff=handoff("left", "right"),
        )

        self.assertEqual(confirming.state, "right_confirming")
        self.assertEqual(confirming.active_side, "left")
        self.assertFalse(confirming.switched_side)
        self.assertEqual(switched.state, "right_active")
        self.assertEqual(switched.active_side, "right")
        self.assertTrue(switched.switched_side)
        self.assertEqual(switched.local_track_id, 3)
        self.assertEqual(coordinator.diagnostics()["handoff_count"], 1)

    def test_strict_handoff_rejects_receiver_outside_entry_roi(self):
        coordinator = CrossHalfBallCoordinator(strict_handoff=True)
        coordinator.update([track(7)], [], timestamp_s=0.0)

        result = coordinator.update(
            [],
            [track(3, center=(700.0, 300.0))],
            timestamp_s=0.02,
            handoff=handoff("left", "right"),
        )

        self.assertEqual(result.active_side, "left")
        self.assertFalse(result.switched_side)
        self.assertEqual(result.state, "left_to_right_armed")
        self.assertEqual(coordinator.diagnostics()["handoff_rejections"], 1)

    def test_strict_handoff_confirmation_resets_when_source_recovers(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            receiver_confirmation_hits=2,
        )
        coordinator.update([track(7)], [], timestamp_s=0.0)
        coordinator.update(
            [],
            [track(3, center=(100.0, 300.0))],
            timestamp_s=0.02,
            handoff=handoff("left", "right"),
        )

        recovered = coordinator.update(
            [track(7, center=(920.0, 300.0))],
            [track(3, center=(110.0, 300.0))],
            timestamp_s=0.04,
            handoff=handoff("left", "right"),
        )

        self.assertEqual(recovered.active_side, "left")
        self.assertEqual(recovered.state, "left_to_right_armed")
        self.assertEqual(coordinator.diagnostics()["confirmation_hits"], 0)

    def test_strict_handoff_window_expires_without_switch(self):
        coordinator = CrossHalfBallCoordinator(strict_handoff=True)
        coordinator.update([track(7)], [], timestamp_s=0.0)
        coordinator.update(
            [track(7)],
            [],
            timestamp_s=0.02,
            handoff=handoff(
                "left",
                "right",
                expires_at_s=0.03,
            ),
        )

        result = coordinator.update(
            [],
            [track(3, center=(100.0, 300.0))],
            timestamp_s=0.05,
        )

        self.assertEqual(result.active_side, "left")
        self.assertFalse(result.switched_side)
        self.assertEqual(coordinator.diagnostics()["handoff_timeouts"], 1)

    def test_strict_handoff_switch_lock_blocks_immediate_reverse_switch(self):
        coordinator = CrossHalfBallCoordinator(
            strict_handoff=True,
            receiver_confirmation_hits=1,
            switch_lock_ms=100.0,
        )
        coordinator.update([track(7)], [], timestamp_s=0.0)

        switched = coordinator.update(
            [track(7, status="predicted", missing_time_ms=40.0)],
            [track(3, center=(100.0, 300.0))],
            timestamp_s=0.02,
            handoff=handoff("left", "right"),
        )
        reverse_attempt = coordinator.update(
            [track(8, center=(900.0, 300.0))],
            [track(3, center=(110.0, 300.0), status="predicted", missing_time_ms=20.0)],
            timestamp_s=0.04,
            handoff=handoff("right", "left"),
        )

        self.assertEqual(switched.active_side, "right")
        self.assertTrue(switched.switched_side)
        self.assertEqual(reverse_attempt.active_side, "right")
        self.assertFalse(reverse_attempt.switched_side)
        self.assertGreater(
            coordinator.diagnostics()["switch_lock_until_s"],
            0.04,
        )

    def test_strict_handoff_suppresses_prediction_past_net_exit(self):
        widths = {"left": 1000, "right": 1000}
        cases = (
            ("left", [track(7, center=(980.0, 300.0))], [], (1010.0, 300.0)),
            ("right", [], [track(7, center=(20.0, 300.0))], (-10.0, 300.0)),
        )
        for side, left_tracks, right_tracks, predicted_center in cases:
            with self.subTest(side=side):
                coordinator = CrossHalfBallCoordinator(strict_handoff=True)
                coordinator.update(
                    left_tracks,
                    right_tracks,
                    timestamp_s=0.0,
                    frame_widths=widths,
                )
                predicted = track(
                    7,
                    status="predicted",
                    missing_time_ms=20.0,
                    center=predicted_center,
                )

                result = coordinator.update(
                    [predicted] if side == "left" else [],
                    [predicted] if side == "right" else [],
                    timestamp_s=0.02,
                    frame_widths=widths,
                    handoff=handoff(
                        side,
                        "right" if side == "left" else "left",
                    ),
                )

                self.assertEqual(result.active_side, side)
                self.assertIsNone(result.track)
                self.assertEqual(
                    coordinator.diagnostics()[
                        "prediction_boundary_rejections"
                    ],
                    1,
                )

    def test_legacy_mode_keeps_existing_boundary_prediction_behaviour(self):
        coordinator = CrossHalfBallCoordinator(strict_handoff=False)
        coordinator.update(
            [track(7, center=(980.0, 300.0))],
            [],
            timestamp_s=0.0,
            frame_widths={"left": 1000, "right": 1000},
        )

        result = coordinator.update(
            [
                track(
                    7,
                    status="predicted",
                    missing_time_ms=20.0,
                    center=(1010.0, 300.0),
                )
            ],
            [],
            timestamp_s=0.02,
            frame_widths={"left": 1000, "right": 1000},
        )

        self.assertIsNotNone(result.track)
        self.assertEqual(
            coordinator.diagnostics()["prediction_boundary_rejections"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
