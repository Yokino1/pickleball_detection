import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tracking.dual_camera.artifacts import DualRunArtifacts
from src.tracking.run_manifest import create_manifest


class RunManifestTest(unittest.TestCase):
    def test_dual_artifact_names_include_manifest_and_partial_variants(self):
        artifacts = DualRunArtifacts.for_run(Path("outputs"), "sample")
        partial = artifacts.partial()

        self.assertEqual(artifacts.manifest.name, "sample_manifest.json")
        self.assertEqual(partial.video.name, "sample_dual_tracking.partial.mp4")
        self.assertEqual(partial.manifest.name, "sample_manifest.partial.json")
        self.assertEqual(len(artifacts.final_files()), 5)

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


if __name__ == "__main__":
    unittest.main()
