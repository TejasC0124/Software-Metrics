import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.profile import LearnerProfileStore


def make_analysis(target, response, weak, scores):
    return {
        "target": target,
        "response": response,
        "weak_mapping": weak,
        "bridge_scores": scores,
    }


class LearnerProfileStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "profiles.json"
        self.store = LearnerProfileStore(self.path)
        self.scores = {"semantic_phonetic": 0.2, "phonetic_visual": 0.8, "semantic_visual": 0.5}

    def tearDown(self):
        self._tmp.cleanup()

    def test_running_strength_is_average_of_observations(self):
        self.store.record_attempt("l1", make_analysis("said", "side", "semantic_phonetic", self.scores))
        profile = self.store.record_attempt(
            "l1", make_analysis("said", "said", None, {"semantic_phonetic": 0.6, "phonetic_visual": 0.8, "semantic_visual": 0.5})
        )
        self.assertEqual(profile["attempts"], 2)
        self.assertAlmostEqual(profile["bridges"]["semantic_phonetic"]["strength"], 0.4)
        self.assertEqual(profile["bridges"]["semantic_phonetic"]["weak_events"], 1)
        self.assertEqual(profile["bridges"]["semantic_phonetic"]["observations"], 2)

    def test_recent_attempts_are_capped_at_twenty(self):
        for _ in range(25):
            profile = self.store.record_attempt("l1", make_analysis("said", "side", None, self.scores))
        self.assertEqual(len(profile["recent_attempts"]), 20)

    def test_corrupt_profile_file_is_treated_as_empty(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        profile = self.store.record_attempt("l1", make_analysis("said", "side", None, self.scores))
        self.assertEqual(profile["attempts"], 1)
        json.loads(self.path.read_text(encoding="utf-8"))

    def test_concurrent_attempts_are_all_saved(self):
        def worker():
            for _ in range(10):
                self.store.record_attempt("shared", make_analysis("said", "side", None, self.scores))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        profile = LearnerProfileStore(self.path).record_attempt(
            "shared", make_analysis("said", "side", None, self.scores)
        )
        self.assertEqual(profile["attempts"], 81)

    def test_blank_learner_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.record_attempt("   ", make_analysis("said", "side", None, self.scores))


if __name__ == "__main__":
    unittest.main()
