import tempfile
import unittest
from pathlib import Path

from src.service import SPBEService


class ServiceTestCase(unittest.TestCase):
    def test_service_returns_analysis_intervention_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SPBEService(Path(directory) / "profiles.json")
            result = service.analyse_attempt("said", "side", "review_learner")
        self.assertIn("analysis", result)
        self.assertIn("intervention", result)
        self.assertEqual(result["profile"]["attempts"], 1)

    def test_service_rejects_blank_values(self):
        with tempfile.TemporaryDirectory() as directory:
            service = SPBEService(Path(directory) / "profiles.json")
            with self.assertRaises(ValueError):
                service.analyse_attempt("", "side", "review_learner")


if __name__ == "__main__":
    unittest.main()
