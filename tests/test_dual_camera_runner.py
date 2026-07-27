import unittest

from src.tracking.dual_camera.coordinator import GlobalBallSelection
from src.tracking.dual_camera.runner import (
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


if __name__ == "__main__":
    unittest.main()
