import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CommandLineTestCase(unittest.TestCase):
    def test_cli_prints_attribution_and_updates_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_file = Path(directory) / "profiles.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--target", "said",
                    "--response", "side",
                    "--learner", "cli_test",
                    "--profile-file", str(profile_file),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Detected weak bridge:", completed.stdout)
        self.assertIn("Profile attempts for cli_test: 1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
