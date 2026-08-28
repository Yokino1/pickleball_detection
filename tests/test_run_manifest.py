import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tracking.dual_camera.artifacts import (
    DualRunArtifacts,
    validate_rally_result_contract,
)
from src.tracking.run_manifest import create_manifest, sha256_json


class RunManifestTest(unittest.TestCase):
    def test_dual_artifacts_are_grouped_in_one_run_directory(self):
        artifacts = DualRunArtifacts.for_run(Path("outputs"), "sample")
        partial = artifacts.partial()

        self.assertEqual(artifacts.run_dir, Path("outputs") / "sample")
        self.assertEqual(artifacts.video.name, "dual_tracking.mp4")
        self.assertEqual(
            artifacts.rally_results_jsonl.name,
            "rally_results.jsonl",
        )
        self.assertEqual(artifacts.manifest.name, "manifest.json")
        self.assertEqual(partial.video.name, "dual_tracking.partial.mp4")
        self.assertEqual(
            partial.rally_results_jsonl.name,
            "rally_results.partial.jsonl",
        )
        self.assertEqual(partial.manifest.name, "manifest.partial.json")
        self.assertTrue(
            all(path.parent == artifacts.run_dir for path in artifacts.final_files())
        )
        self.assertTrue(
            all(path.parent == artifacts.run_dir for path in artifacts.all_files())
        )
        self.assertNotIn(
            artifacts.rally_results_jsonl,
            artifacts.final_files(),
        )
        self.assertIn(
            artifacts.rally_results_jsonl,
            artifacts.all_files(),
        )
        self.assertEqual(
            artifacts.required_files(referee_enabled=False),
            artifacts.final_files(),
        )
        self.assertEqual(
            artifacts.required_files(referee_enabled=True),
            artifacts.all_files(),
        )
        self.assertEqual(len(artifacts.final_files()), 5)
        self.assertEqual(len(artifacts.all_files()), 6)

    def test_optional_rally_results_do_not_break_legacy_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = DualRunArtifacts.for_run(Path(directory), "legacy")
            artifacts.run_dir.mkdir()
            partial = artifacts.partial()
            for path in partial.final_files():
                path.write_text(path.name, encoding="utf-8")

            artifacts.promote_from(partial)

            self.assertTrue(
                all(path.is_file() for path in artifacts.final_files())
            )
            self.assertFalse(artifacts.rally_results_jsonl.exists())

    def test_rally_results_are_promoted_and_removed_with_run_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = DualRunArtifacts.for_run(Path(directory), "demo")
            artifacts.run_dir.mkdir()
            partial = artifacts.partial()
            for path in partial.final_files():
                path.write_text(path.name, encoding="utf-8")
            partial.rally_results_jsonl.write_text(
                '{"rally_id": 1}\n',
                encoding="utf-8",
            )

            artifacts.promote_from(partial)

            self.assertEqual(
                artifacts.rally_results_jsonl.read_text(encoding="utf-8"),
                '{"rally_id": 1}\n',
            )
            artifacts.remove()
            self.assertFalse(
                any(path.exists() for path in artifacts.all_files())
            )

    def test_promotion_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = DualRunArtifacts.for_run(Path(directory), "protected")
            artifacts.run_dir.mkdir()
            partial = artifacts.partial()
            for path in partial.final_files():
                path.write_text(path.name, encoding="utf-8")
            artifacts.video.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                artifacts.promote_from(partial)

            self.assertEqual(
                artifacts.video.read_text(encoding="utf-8"),
                "keep",
            )

    def test_dual_artifact_run_id_cannot_escape_output_directory(self):
        for run_id in ("", ".", "..", "../outside", r"..\outside"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    DualRunArtifacts.for_run(Path("outputs"), run_id)

    def test_manifest_records_profile_and_config_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "tracking.yaml"
            config_path.write_text(
                "schema_version: 1\n"
                "profile:\n"
                "  name: test\n"
                "  revision: 3\n",
                encoding="utf-8",
            )
            config = {
                "schema_version": 1,
                "profile": {"name": "test", "revision": 3},
                "tracker": {"max_output_tracks": 0},
                "runtime": {
                    "court_projection": {"enabled": False},
                    "person_detection": {
                        "player_selection": {"max_players": 2}
                    },
                },
            }

            with patch(
                "src.tracking.run_manifest.collect_git_state",
                return_value={"available": False},
            ):
                manifest = create_manifest(
                    project_root=root,
                    run_id="run-1",
                    run_type="dual_camera_tracking",
                    config_path=config_path,
                    config=config,
                    inputs=[],
                    parameters={"max_frames": 30},
                )

            self.assertEqual(manifest["run_id"], "run-1")
            self.assertEqual(manifest["config"]["profile"]["revision"], 3)
            self.assertEqual(len(manifest["config"]["sha256"]), 64)
            self.assertEqual(
                manifest["config"]["effective"],
                config,
            )
            self.assertEqual(
                manifest["config"]["effective_sha256"],
                sha256_json(config),
            )
            config["tracker"]["max_output_tracks"] = 1
            self.assertEqual(
                manifest["config"]["effective"]["tracker"][
                    "max_output_tracks"
                ],
                0,
            )

    def test_rally_stream_matches_terminal_snapshots_and_final_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_jsonl = root / "global.jsonl"
            rally_jsonl = root / "rally.jsonl"
            result = {
                "rally_id": 1,
                "server_side": "left",
                "rally_winner": "left",
                "status": "demo_inferred",
                "next_server_side": "left",
                "point_awarded_to": "left",
                "score_before": {"left": 0, "right": 0},
                "score_after": {"left": 1, "right": 0},
            }
            records = [
                {
                    "frame_index": 0,
                    "referee": {
                        "score": {"left": 0, "right": 0},
                        "service_owner_side": "left",
                        "rally_result": None,
                    },
                },
                {
                    "frame_index": 1,
                    "referee": {
                        "score": {"left": 1, "right": 0},
                        "service_owner_side": "left",
                        "rally_result": result,
                    },
                },
            ]
            global_jsonl.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            rally_jsonl.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )

            diagnostics = validate_rally_result_contract(
                global_jsonl,
                rally_jsonl,
            )

            self.assertTrue(diagnostics["enabled"])
            self.assertEqual(diagnostics["rally_results"], 1)
            self.assertEqual(
                diagnostics["final_score"],
                {"left": 1, "right": 0},
            )
            self.assertEqual(
                diagnostics["final_service_owner_side"],
                "left",
            )

    def test_rally_stream_rejects_next_server_that_is_not_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_jsonl = root / "global.jsonl"
            rally_jsonl = root / "rally.jsonl"
            result = {
                "rally_id": 1,
                "server_side": "left",
                "rally_winner": "right",
                "status": "demo_inferred",
                "next_server_side": "left",
                "point_awarded_to": "right",
                "score_before": {"left": 0, "right": 0},
                "score_after": {"left": 0, "right": 1},
            }
            global_jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "referee": {
                            "score": {"left": 0, "right": 1},
                            "service_owner_side": "left",
                            "rally_result": result,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rally_jsonl.write_text(
                json.dumps(result) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "winner must own the next serve",
            ):
                validate_rally_result_contract(global_jsonl, rally_jsonl)

    def test_rally_stream_rejects_result_that_differs_from_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_jsonl = root / "global.jsonl"
            rally_jsonl = root / "rally.jsonl"
            result = {
                "rally_id": 1,
                "point_awarded_to": None,
                "score_before": {"left": 0, "right": 0},
                "score_after": {"left": 0, "right": 0},
            }
            global_jsonl.write_text(
                json.dumps(
                    {
                        "frame_index": 0,
                        "referee": {
                            "score": {"left": 0, "right": 0},
                            "rally_result": result,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rally_jsonl.write_text(
                json.dumps({**result, "rally_id": 2}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_rally_result_contract(global_jsonl, rally_jsonl)

    def test_legacy_global_stream_without_referee_needs_no_rally_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_jsonl = root / "global.jsonl"
            global_jsonl.write_text(
                '{"frame_index": 0}\n',
                encoding="utf-8",
            )

            diagnostics = validate_rally_result_contract(
                global_jsonl,
                root / "missing_rally.jsonl",
            )

            self.assertFalse(diagnostics["enabled"])


if __name__ == "__main__":
    unittest.main()
