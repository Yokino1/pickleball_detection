import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.tracking.dual_camera.projection_replay import (
    ProjectionReplayArtifacts,
    ProjectionReplaySource,
    _ball_track_from_record,
    _court_motion_discontinuity_reason,
    _record_timestamp_s,
    _remove_failed_replay,
    load_projection_replay_source,
    replay_court_projection,
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
        self.assertEqual(
            artifacts.rally_results_jsonl.name,
            "rally_results.jsonl",
        )
        self.assertEqual(
            artifacts.partial_rally_results_jsonl().name,
            "rally_results.partial.jsonl",
        )
        self.assertIn(artifacts.rally_results_jsonl, artifacts.all_paths())
        self.assertIn(
            artifacts.partial_rally_results_jsonl(),
            artifacts.all_paths(),
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

    def test_projection_only_source_does_not_require_rendered_video(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
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

            source = load_projection_replay_source(
                run_dir,
                require_rendered_video=False,
            )

            self.assertFalse(source.rendered_video.exists())
            self.assertTrue(source.global_jsonl.is_file())

    def test_source_loader_rejects_truncated_completed_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "left_tracking.jsonl",
                "right_tracking.jsonl",
                "global_tracking.jsonl",
            ):
                (run_dir / name).write_text(
                    '{"frame_index": 0}\n{"frame_index": 1}\n',
                    encoding="utf-8",
                )
            manifest = {
                "run_id": "source",
                "status": "completed",
                "inputs": [
                    {
                        "side": "left",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                        "frames": 3,
                    },
                    {
                        "side": "right",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                        "frames": 3,
                    },
                ],
                "parameters": {"header_height": 82},
                "outputs": {},
                "summary": {"source_fps": 50.0, "processed_frames": 3},
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "does not match completed manifest",
            ):
                load_projection_replay_source(
                    run_dir,
                    require_rendered_video=False,
                )

    def test_source_loader_rejects_non_contiguous_frame_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "left_tracking.jsonl",
                "right_tracking.jsonl",
                "global_tracking.jsonl",
            ):
                (run_dir / name).write_text(
                    '{"frame_index": 0}\n{"frame_index": 2}\n',
                    encoding="utf-8",
                )
            manifest = {
                "run_id": "source",
                "status": "completed",
                "inputs": [
                    {
                        "side": "left",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                        "frames": 2,
                    },
                    {
                        "side": "right",
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                        "frames": 2,
                    },
                ],
                "parameters": {"header_height": 82},
                "outputs": {},
                "summary": {"source_fps": 50.0, "processed_frames": 2},
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "contiguous from zero"):
                load_projection_replay_source(
                    run_dir,
                    require_rendered_video=False,
                )

    def test_source_loader_records_validated_file_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for name in (
                "left_tracking.jsonl",
                "right_tracking.jsonl",
                "global_tracking.jsonl",
            ):
                (run_dir / name).write_text(
                    '{"frame_index": 0}\n',
                    encoding="utf-8",
                )
            manifest = {
                "run_id": "source",
                "status": "completed",
                "inputs": [
                    {
                        "side": side,
                        "width": 160,
                        "height": 180,
                        "fps": 50.0,
                        "frames": 1,
                    }
                    for side in ("left", "right")
                ],
                "parameters": {"header_height": 82},
                "outputs": {},
                "summary": {"source_fps": 50.0, "processed_frames": 1},
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            source = load_projection_replay_source(
                run_dir,
                require_rendered_video=False,
            )

            self.assertEqual(source.expected_frames, 1)
            self.assertEqual(
                source.artifact_identities["global_jsonl"]["bytes"],
                (run_dir / "global_tracking.jsonl").stat().st_size,
            )
            self.assertEqual(
                len(source.artifact_identities["global_jsonl"]["sha256"]),
                64,
            )
            self.assertEqual(len(source.source_manifest_sha256), 64)

    def test_failed_replay_removes_new_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            source_dir.mkdir()
            for name in (
                "left_tracking.jsonl",
                "right_tracking.jsonl",
                "global_tracking.jsonl",
            ):
                (source_dir / name).touch()
            source = ProjectionReplaySource(
                run_dir=source_dir,
                run_id="source",
                rendered_video=source_dir / "missing.mp4",
                left_jsonl=source_dir / "left_tracking.jsonl",
                right_jsonl=source_dir / "right_tracking.jsonl",
                global_jsonl=source_dir / "global_tracking.jsonl",
                left_size=(160, 180),
                right_size=(160, 180),
                fps=50.0,
                header_height=82,
                frame_scale_overrides={"left": 0.25, "right": 0.25},
                manifest={},
            )
            renderer = Mock()
            renderer.panel_width.return_value = 320
            writer = Mock()
            writer.isOpened.return_value = True
            output_dir = root / "replays"
            config_path = root / "configs" / "tracking.yaml"
            config_path.parent.mkdir()
            config_path.write_text("{}\n", encoding="utf-8")

            with (
                patch(
                    "src.tracking.dual_camera.projection_replay."
                    "build_court_projection",
                    return_value=(Mock(), renderer, Mock()),
                ),
                patch(
                    "src.tracking.dual_camera.projection_replay."
                    "cv2.VideoWriter",
                    return_value=writer,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "produced no frames",
                ):
                    replay_court_projection(
                        source,
                        output_dir=output_dir,
                        run_id="failed_replay",
                        config={"output": {"codec": "mp4v"}},
                        config_path=config_path,
                        projection_only=True,
                    )

            self.assertFalse((output_dir / "failed_replay").exists())

    def test_failed_replay_cleanup_removes_rally_result_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = ProjectionReplayArtifacts.for_run(
                Path(directory),
                "failed",
            )
            artifacts.run_dir.mkdir()
            for path in artifacts.all_paths():
                path.touch()

            _remove_failed_replay(artifacts)

            self.assertFalse(artifacts.run_dir.exists())
            self.assertFalse(
                any(path.exists() for path in artifacts.all_paths())
            )

    def test_record_timestamp_prefers_valid_saved_value(self):
        self.assertEqual(
            _record_timestamp_s(
                {"timestamp": "1.25"},
                frame_index=50,
                source_fps=50.0,
            ),
            1.25,
        )

    def test_record_timestamp_falls_back_to_frame_clock(self):
        for saved_value in (None, "bad", float("nan"), -1.0):
            with self.subTest(saved_value=saved_value):
                record = (
                    {}
                    if saved_value is None
                    else {"timestamp": saved_value}
                )
                self.assertEqual(
                    _record_timestamp_s(
                        record,
                        frame_index=75,
                        source_fps=50.0,
                    ),
                    1.5,
                )

    def test_display_only_reset_does_not_clear_replayed_court_motion(self):
        self.assertIsNone(
            _court_motion_discontinuity_reason("camera_side_switch")
        )
        self.assertIsNone(_court_motion_discontinuity_reason("observation_gap"))
        self.assertEqual(
            _court_motion_discontinuity_reason("physical_discontinuity"),
            "physical_discontinuity",
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
