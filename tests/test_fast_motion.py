import unittest

import numpy as np

from src.tracking.fast_motion import FastMotionProposalGenerator


class FastMotionProposalGeneratorTest(unittest.TestCase):
    def test_frame_scale_override_does_not_shrink_after_crop(self):
        generator = FastMotionProposalGenerator(
            reference_frame_width=1280,
            frame_scale_override=3.0,
        )

        self.assertEqual(generator.frame_scale(1920), 3.0)

    def test_emits_only_after_consecutive_fast_motion_inside_roi(self):
        generator = FastMotionProposalGenerator(
            analysis_width=160,
            difference_threshold=10,
            min_component_area=1,
            max_component_area=100,
            max_component_dimension=20,
            min_speed_px_per_second=1,
            max_speed_px_per_second=2000,
            min_streak=2,
            blur_kernel=1,
            dilation_iterations=0,
        )
        roi = [0.0, 0.0, 80.0, 80.0]

        def frame_at(x):
            frame = np.zeros((80, 160, 3), dtype=np.uint8)
            frame[30:34, x:x + 4] = 255
            return frame

        first = generator.propose(
            frame_at(10),
            timestamp_s=0.0,
            search_roi=roi,
        )
        second = generator.propose(
            frame_at(20),
            timestamp_s=1 / 30,
            search_roi=roi,
        )
        third = generator.propose(
            frame_at(30),
            timestamp_s=2 / 30,
            search_roi=roi,
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertTrue(third)
        self.assertTrue(all(item.source == "fast_motion" for item in third))

    def test_stays_dormant_without_search_roi(self):
        generator = FastMotionProposalGenerator(analysis_width=160)
        frame = np.zeros((80, 160, 3), dtype=np.uint8)

        generator.propose(frame, timestamp_s=0.0, search_roi=None)
        proposals = generator.propose(
            np.full_like(frame, 255),
            timestamp_s=1 / 30,
            search_roi=None,
        )

        self.assertEqual(proposals, [])
        self.assertFalse(generator.diagnostics["enabled"])


if __name__ == "__main__":
    unittest.main()
