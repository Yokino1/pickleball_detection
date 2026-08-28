import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2

from src.court.renderer import CourtPanelRenderer
from src.tracking.dual_camera.coordinator import GlobalBallSelection
from src.tracking.dual_camera.runner import (
    _court_motion_discontinuity_reason,
    _global_frame_record,
    _read_and_validate_metadata,
    _remember_rendered_track,
    _stream_frame_scale_overrides,
    _trail_reset_reason,
)
from src.tracking.types import BallTrack


def selection(
    *,
    side="left",
    local_track_id=1,
    center=(100.0, 100.0),
    switched_side=False,
):
    return GlobalBallSelection(
        global_track_id=1,
        active_side=side,
        state=f"{side}_active",
        track=BallTrack(
            track_id=1,
            status="observed",
            center=list(center),
            confidence=0.8,
            source="detector",
            observation_source="yolo",
        ),
        local_track_id=local_track_id,
        switched_side=switched_side,
    )


class DualCameraRunnerTest(unittest.TestCase):
    class Capture:
        def __init__(self, *, fps=50.0, frames=100, width=640, height=720):
            self.values = {
                cv2.CAP_PROP_FPS: fps,
                cv2.CAP_PROP_FRAME_COUNT: frames,
                cv2.CAP_PROP_FRAME_WIDTH: width,
                cv2.CAP_PROP_FRAME_HEIGHT: height,
            }

        def get(self, property_id):
            return self.values[property_id]

    def test_video_metadata_rejects_nonfinite_fps(self):
        with self.assertRaisesRegex(ValueError, "Invalid left FPS"):
            _read_and_validate_metadata(
                self.Capture(fps=float("nan")),
                self.Capture(),
            )

    def test_video_metadata_rejects_nonpositive_geometry(self):
        with self.assertRaisesRegex(ValueError, "Invalid right video width"):
            _read_and_validate_metadata(
                self.Capture(),
                self.Capture(width=0),
            )

    def test_paired_crop_scale_uses_total_source_width(self):
        config = {
            "tracker": {"reference_frame_width": 1280.0},
            "runtime": {
                "dual_camera_streams": {
                    "frame_scale_mode": "paired_crop_total_width",
                    "left": {},
                    "right": {},
                }
            },
        }

        self.assertEqual(
            _stream_frame_scale_overrides(
                config,
                {"left": 640, "right": 640},
            ),
            {"left": 1.0, "right": 1.0},
        )
        self.assertEqual(
            _stream_frame_scale_overrides(
                config,
                {"left": 618, "right": 662},
            ),
            {"left": 1.0, "right": 1.0},
        )
        self.assertEqual(
            _stream_frame_scale_overrides(
                config,
                {"left": 1920, "right": 1920},
            ),
            {"left": 3.0, "right": 3.0},
        )

    def test_explicit_stream_scale_overrides_inferred_pair_scale(self):
        config = {
            "tracker": {"reference_frame_width": 1280.0},
            "runtime": {
                "dual_camera_streams": {
                    "frame_scale_mode": "paired_crop_total_width",
                    "left": {"frame_scale_override": 1.25},
                    "right": {},
                }
            },
        }

        self.assertEqual(
            _stream_frame_scale_overrides(
                config,
                {"left": 640, "right": 640},
            ),
            {"left": 1.25, "right": 1.0},
        )

    def test_trail_resets_when_local_track_changes(self):
        previous = {
            "left": (3, (100.0, 100.0), 0.0),
            "right": None,
        }

        reason = _trail_reset_reason(
            selection(local_track_id=4, center=(105.0, 100.0)),
            timestamp_s=0.05,
            previous_rendered_tracks=previous,
            max_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 1.0, "right": 1.0},
        )

        self.assertEqual(reason, "local_track_change")

    def test_trail_resets_on_impossible_same_track_correction(self):
        previous = {
            "left": (3, (100.0, 100.0), 0.0),
            "right": None,
        }

        reason = _trail_reset_reason(
            selection(local_track_id=3, center=(300.0, 100.0)),
            timestamp_s=0.05,
            previous_rendered_tracks=previous,
            max_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 1.0, "right": 1.0},
        )

        self.assertEqual(reason, "physical_discontinuity")

    def test_continuous_detection_keeps_trail(self):
        previous = {"left": None, "right": None}
        current = selection(local_track_id=3, center=(100.0, 100.0))
        _remember_rendered_track(current, 0.0, previous)

        reason = _trail_reset_reason(
            selection(local_track_id=3, center=(140.0, 100.0)),
            timestamp_s=0.05,
            previous_rendered_tracks=previous,
            max_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 1.0, "right": 1.0},
        )

        self.assertIsNone(reason)

    def test_missing_frame_preserves_identity_for_gap_reset(self):
        previous = {"left": None, "right": None}
        current = selection(local_track_id=3, center=(100.0, 100.0))
        _remember_rendered_track(current, 0.0, previous)
        _remember_rendered_track(
            SimpleNamespace(track=None, active_side="left"),
            0.02,
            previous,
        )

        reason = _trail_reset_reason(
            selection(local_track_id=3, center=(110.0, 100.0)),
            timestamp_s=0.04,
            previous_rendered_tracks=previous,
            max_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 1.0, "right": 1.0},
            max_observation_gap_s=0.03,
        )

        self.assertEqual(reason, "observation_gap")

    def test_track_change_after_gap_is_still_detected(self):
        previous = {"left": None, "right": None}
        _remember_rendered_track(
            selection(local_track_id=3, center=(100.0, 100.0)),
            0.0,
            previous,
        )
        _remember_rendered_track(
            SimpleNamespace(track=None, active_side="left"),
            0.02,
            previous,
        )

        reason = _trail_reset_reason(
            selection(local_track_id=4, center=(110.0, 100.0)),
            timestamp_s=0.04,
            previous_rendered_tracks=previous,
            max_speed_px_per_second=3200.0,
            frame_scale_overrides={"left": 1.0, "right": 1.0},
            max_observation_gap_s=0.03,
        )

        self.assertEqual(reason, "local_track_change")

    def test_display_only_resets_do_not_clear_court_motion(self):
        self.assertIsNone(
            _court_motion_discontinuity_reason("camera_side_switch")
        )
        self.assertIsNone(_court_motion_discontinuity_reason("observation_gap"))
        self.assertEqual(
            _court_motion_discontinuity_reason("local_track_change"),
            "local_track_change",
        )

    def test_global_record_exposes_referee_as_top_level_contract(self):
        referee = {
            "schema_version": 1,
            "phase": "IN_RALLY",
            "score": {"left": 2, "right": 1},
            "rally_result": None,
        }
        coordinator = Mock()
        coordinator.diagnostics.return_value = {"handoff_count": 0}
        left_result = SimpleNamespace(
            ball_tracks=[],
            players=[SimpleNamespace(eligible_player=True)],
        )
        right_result = SimpleNamespace(ball_tracks=[], players=[])

        record = _global_frame_record(
            12,
            0.24,
            selection(),
            coordinator,
            None,
            left_result,
            right_result,
            referee=referee,
        )

        self.assertEqual(record["referee"], referee)
        self.assertIsNone(record["court"])
        self.assertNotIn("referee", record["global_ball"])
        self.assertEqual(record["left"]["eligible_player_count"], 1)

    def test_global_record_keeps_referee_key_when_feature_is_disabled(self):
        coordinator = Mock()
        coordinator.diagnostics.return_value = {}
        result = SimpleNamespace(ball_tracks=[], players=[])

        record = _global_frame_record(
            0,
            0.0,
            selection(),
            coordinator,
            None,
            result,
            result,
        )

        self.assertIn("referee", record)
        self.assertIsNone(record["referee"])

    def test_renderer_latches_one_shot_rally_result_for_two_seconds(self):
        renderer = CourtPanelRenderer(referee_result_hold_seconds=2.0)
        result = {
            "rally_id": 4,
            "terminal_event": "second_bounce",
            "rally_winner": "left",
        }

        self.assertEqual(
            renderer._visible_rally_result(
                {
                    "timestamp_s": 10.0,
                    "rally_id": 4,
                    "rally_result": result,
                }
            ),
            result,
        )
        self.assertEqual(
            renderer._visible_rally_result(
                {
                    "timestamp_s": 11.5,
                    "rally_id": 4,
                    "rally_result": None,
                }
            ),
            result,
        )
        self.assertIsNone(
            renderer._visible_rally_result(
                {
                    "timestamp_s": 12.01,
                    "rally_id": None,
                    "rally_result": None,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
