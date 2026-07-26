import unittest

from src.tracking.person_tracking import PersonBoxTracker, PlayerSelector
from src.tracking.types import PlayerDetection


def person(bbox, confidence=0.9):
    return PlayerDetection(
        bbox=list(bbox),
        confidence=confidence,
        foot_point=[(bbox[0] + bbox[2]) / 2.0, bbox[3]],
    )


class PersonTrackingTest(unittest.TestCase):
    def test_box_is_retained_between_detector_runs(self):
        tracker = PersonBoxTracker(default_fps=25)
        first = tracker.update([person([100, 50, 160, 210])], 640, 360, 0.0)
        skipped = tracker.update(None, 640, 360, 0.04)

        self.assertEqual(first[0].track_id, skipped[0].track_id)
        self.assertEqual(skipped[0].source, "person_track")

    def test_selector_excludes_spectator_region(self):
        selector = PlayerSelector(
            max_players=2,
            play_area_normalized=[0.0, 0.2, 1.0, 1.0],
            spectator_exclusion_regions=[[0.0, 0.0, 0.4, 0.7]],
        )
        people = [
            person([20, 20, 100, 200]),
            person([400, 80, 500, 330]),
        ]
        people[0].track_id = 1
        people[1].track_id = 2

        selected = selector.select(people, 640, 360)

        self.assertFalse(selected[0].eligible_player)
        self.assertTrue(selected[1].eligible_player)


if __name__ == "__main__":
    unittest.main()
