import unittest

from src.cross_modal import CrossModalEngine


class CrossModalAttributionTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = CrossModalEngine()

    def test_bridge_score_is_mean_of_its_endpoints(self):
        scores = self.engine.calculate_bridge_scores(
            {"semantic": 0.2, "phonetic": 0.6, "visual": 1.0}
        )
        self.assertEqual(scores["semantic_phonetic"], 0.4)
        self.assertEqual(scores["phonetic_visual"], 0.8)
        self.assertEqual(scores["semantic_visual"], 0.6)

    def test_missing_modality_disables_its_bridges(self):
        scores = self.engine.calculate_bridge_scores(
            {"semantic": 0.5, "phonetic": None, "visual": 0.9}
        )
        self.assertIsNone(scores["semantic_phonetic"])
        self.assertIsNone(scores["phonetic_visual"])
        self.assertEqual(scores["semantic_visual"], 0.7)

    def test_lowest_bridge_is_selected(self):
        analysis = self.engine.analyze_error("said", "side", 0.2, 0.9, 0.95)
        self.assertEqual(analysis["weak_mapping"], "semantic_phonetic")
        self.assertFalse(analysis["is_correct"])

    def test_correct_response_has_no_weak_bridge(self):
        analysis = self.engine.analyze_error("said", "said", 1.0, 1.0, 1.0)
        self.assertTrue(analysis["is_correct"])
        self.assertIsNone(analysis["weak_mapping"])
        self.assertEqual(analysis["attribution_confidence"], 0.0)

    def test_scores_are_clamped_to_unit_interval(self):
        scores = self.engine.calculate_cross_modal_scores(1.7, -0.3, "not a number")
        self.assertEqual(scores, {"semantic": 1.0, "phonetic": 0.0, "visual": None})

    def test_confidence_is_gap_between_two_lowest_bridges(self):
        analysis = self.engine.analyze_error("said", "side", 0.2, 0.6, 1.0)
        # Bridges: semantic_phonetic 0.4, phonetic_visual 0.8, semantic_visual 0.6
        self.assertAlmostEqual(analysis["attribution_confidence"], 0.2)

    def test_no_valid_bridges_gives_no_weak_mapping(self):
        self.assertIsNone(self.engine.identify_weak_mapping({"a": None}))


if __name__ == "__main__":
    unittest.main()
