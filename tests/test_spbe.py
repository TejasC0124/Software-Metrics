import tempfile
import unittest
from pathlib import Path

from src.cross_modal import CrossModalEngine
from src.intervention import InterventionEngine
from src.phonetic import PhoneticEngine
from src.profile import LearnerProfileStore
from src.semantic import SemanticEngine
from src.visual import VisualEngine


class SPBETestCase(unittest.TestCase):
    def setUp(self):
        self.semantic = SemanticEngine()
        self.phonetic = PhoneticEngine()
        self.visual = VisualEngine()
        self.cross_modal = CrossModalEngine()

    def test_text_response_produces_three_bridge_scores(self):
        analysis = self.cross_modal.analyze_text_response(
            "said", "side", self.semantic, self.phonetic, self.visual
        )
        self.assertFalse(analysis["is_correct"])
        self.assertEqual(set(analysis["bridge_scores"]), set(CrossModalEngine.BRIDGES))
        self.assertIn(analysis["weak_mapping"], CrossModalEngine.BRIDGES)
        self.assertTrue(
            all(0.0 <= score <= 1.0 for score in analysis["modality_scores"].values())
        )

    def test_correct_response_needs_no_remedial_intervention(self):
        analysis = self.cross_modal.analyze_text_response(
            "said", "said", self.semantic, self.phonetic, self.visual
        )
        intervention = InterventionEngine().generate_intervention(
            analysis["weak_mapping"], "said", "said"
        )
        self.assertTrue(analysis["is_correct"])
        self.assertIsNone(analysis["weak_mapping"])
        self.assertEqual(intervention["title"], "Positive reinforcement")

    def test_profile_records_bridge_observations(self):
        analysis = self.cross_modal.analyze_text_response(
            "said", "side", self.semantic, self.phonetic, self.visual
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = LearnerProfileStore(Path(directory) / "profiles.json").record_attempt(
                "learner_a", analysis
            )
        self.assertEqual(profile["attempts"], 1)
        self.assertEqual(
            profile["bridges"][analysis["weak_mapping"]]["weak_events"], 1
        )


if __name__ == "__main__":
    unittest.main()
