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
        hits=5,
        confirmed=True,
        motion_confirmed=True,
    )


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


if __name__ == "__main__":
    unittest.main()
