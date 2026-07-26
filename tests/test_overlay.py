import unittest

import numpy as np

from src.tracking.overlay import TrackingOverlay
from src.tracking.types import BallTrack, FrameResult, PlayerDetection


class TrackingOverlayTest(unittest.TestCase):
    def test_player_boxes_can_be_hidden_without_removing_player_data(self):
        overlay = TrackingOverlay(draw_players=False)
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        result = FrameResult(
            frame_index=1,
            width=180,
            height=120,
            players=[
                PlayerDetection(
                    bbox=[20.0, 40.0, 80.0, 110.0],
                    track_id=7,
                    eligible_player=True,
                )
            ],
        )

        drawn = overlay.draw(frame, result)

        self.assertEqual(drawn[40, 20].tolist(), [0, 0, 0])
        self.assertEqual(len(result.players), 1)
        self.assertTrue(result.players[0].eligible_player)

    def test_every_track_id_uses_fluorescent_green(self):
        overlay = TrackingOverlay(trail_length=3)
        for track_id in (1, 2, 7):
            frame = np.zeros((120, 180, 3), dtype=np.uint8)
            result = FrameResult(
                frame_index=track_id,
                width=180,
                height=120,
                ball_tracks=[
                    BallTrack(
                        track_id=track_id,
                        status="observed",
                        center=[80.0, 60.0],
                        bbox=[74.0, 54.0, 86.0, 66.0],
                        confidence=0.9,
                    )
                ],
            )

            drawn = overlay.draw(frame, result)

            self.assertEqual(drawn[60, 80].tolist(), [0, 255, 0])


if __name__ == "__main__":
    unittest.main()
