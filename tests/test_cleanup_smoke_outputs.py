import os
import tempfile
import unittest
from pathlib import Path

from tools.cleanup_smoke_outputs import find_expired_runs


class CleanupSmokeOutputsTest(unittest.TestCase):
    def test_lists_only_expired_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_run = root / "old_run"
            new_run = root / "new_run"
            old_run.mkdir()
            new_run.mkdir()
            (root / "note.txt").write_text("keep", encoding="utf-8")
            os.utime(old_run, (100.0, 100.0))
            os.utime(new_run, (900.0, 900.0))

            expired = find_expired_runs(
                root,
                older_than_days=500.0 / 86400.0,
                now_s=1000.0,
            )

            self.assertEqual(expired, [old_run])


if __name__ == "__main__":
    unittest.main()
