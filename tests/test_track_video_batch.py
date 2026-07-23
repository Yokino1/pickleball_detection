import tempfile
import unittest
from pathlib import Path

from apps.track_video import batch_output_paths, discover_videos


class TrackVideoBatchTest(unittest.TestCase):
    def test_discovers_supported_videos_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "b.MP4").touch()
            (root / "a.mp4").touch()
            (root / "notes.txt").touch()
            nested = root / "session"
            nested.mkdir()
            (nested / "c.mov").touch()

            self.assertEqual([path.name for path in discover_videos(root)], ["a.mp4", "b.MP4"])
            self.assertEqual(
                [path.name for path in discover_videos(root, recursive=True)],
                ["a.mp4", "b.MP4", "c.mov"],
            )

    def test_batch_outputs_preserve_relative_directory(self):
        input_root = Path("input")
        video = input_root / "session_01" / "match.mp4"
        output_video, output_jsonl = batch_output_paths(video, input_root, Path("outputs"))

        self.assertEqual(output_video, Path("outputs/session_01/match_tracked.mp4"))
        self.assertEqual(output_jsonl, Path("outputs/session_01/match_tracking.jsonl"))


if __name__ == "__main__":
    unittest.main()
