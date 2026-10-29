import unittest

from src.intervention import InterventionEngine


class InterventionTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = InterventionEngine()

    def test_every_bridge_has_a_three_step_intervention(self):
        for bridge in ("semantic_phonetic", "phonetic_visual", "semantic_visual"):
            with self.subTest(bridge=bridge):
                plan = self.engine.generate_intervention(bridge, "said", "side")
                self.assertTrue(plan["title"])
                self.assertEqual(len(plan["activities"]), 3)
                self.assertEqual(plan["weak_mapping"], bridge)
                self.assertEqual(plan["target"], "said")

    def test_no_weak_bridge_gives_positive_reinforcement(self):
        plan = self.engine.generate_intervention(None, "said", "said")
        self.assertEqual(plan["title"], "Positive reinforcement")

    def test_unknown_bridge_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.generate_intervention("visual_only", "said", "side")


if __name__ == "__main__":
    unittest.main()
