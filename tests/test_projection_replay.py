import json
import tempfile
import unittest
from pathlib import Path

from src.tracking.dual_camera.projection_replay import (
    ProjectionReplayArtifacts,
    _ball_track_from_record,
    load_projection_replay_source,
)


class ProjectionReplayTest(unittest.TestCase):
    def test_replay_artifacts_are_grouped_and_cannot_escape(self):
        artifacts = ProjectionReplayArtifacts.for_run(
            Path("outputs"),
            "sample",
        )

        self.assertEqual(artifacts.run_dir, Path("outputs") / "sample")
        self.assertEqual(artifacts.video.name, "projection_replay.mp4")
        self.assertEqual(
            artifacts.global_jsonl.name,
            "global_projection.jsonl",
        )
        for run_id in ("", ".", "..", "../outside", r"..\outside"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    ProjectionReplayArtifacts.for_run(
                        Path("outputs"),
                        run_id,
                    )

    def test_saved_track_reconstruction_ignores_future_fields(self):
        track = _ball_track_from_record(
            {
                "track_id": 7,
                "status": "observed",
                "center": [100.0, 120.0],
                "velocity": [300.0, -200.0],
                "future_field": "ignored",
            }
        )

        self.assertIsNotNone(track)
        self.assertEqual(track.track_id, 7)
        self.assertEqual(track.center, [100.0, 120.0])
        self.assertEqual(track.velocity, [300.0, -200.0])

    def test_source_loader_uses_saved_geometry_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "dual_tracking.mp4",
                "left_tracking.jsonl",
                "right_tracking.jsonl",
                "global_tracking.jsonl",
            ):
                (run_dir / name).touch()
            manifest = {
                "run_id": "source",
                "status": "completed",
                "inputs": [
                    {
                        "side": "left",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                    },
                    {
                        "side": "right",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                    },
                ],
                "parameters": {"header_height": 82},
                "outputs": {},
                "summary": {"source_fps": 50.0},
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            source = load_projection_replay_source(run_dir)

            self.assertEqual(source.tracking_width, 320)
            self.assertEqual(source.frame_height, 180)
            self.assertEqual(source.fps, 50.0)
            self.assertEqual(
                source.frame_scale_overrides,
                {"left": 0.25, "right": 0.25},
            )

    def test_replay_module_does_not_build_detection_or_tracking_models(self):
        module_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "tracking"
            / "dual_camera"
            / "projection_replay.py"
        )
        source = module_path.read_text(encoding="utf-8")

        for forbidden in (
            "build_detector",
            "build_person_detector",
            "build_pipeline",
            "MultiBallTracker(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
