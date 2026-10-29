import unittest
from unittest import mock

from src.semantic import SemanticEngine, WORDNET_SETUP_MESSAGE


class SemanticEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticEngine()

    def test_identical_words_score_one(self):
        self.assertEqual(self.engine.semantic_similarity("said", "said"), 1.0)

    def test_related_forms_share_meaning(self):
        # "say" is the base form of "said" and shares a WordNet synset with it.
        self.assertEqual(self.engine.semantic_similarity("said", "say"), 1.0)

    def test_unknown_word_scores_zero(self):
        self.assertEqual(self.engine.semantic_similarity("said", "qzxvw"), 0.0)

    def test_process_word_without_embedding_returns_definitions(self):
        result = self.engine.process_word("said")
        self.assertTrue(result["definitions"])
        self.assertIsNone(result["embedding"])

    def test_embedding_uses_loaded_model(self):
        fake_model = mock.Mock()
        fake_model.encode.return_value = [0.1, 0.2, 0.3]
        with mock.patch.object(SemanticEngine, "_load_embedding_model", return_value=fake_model):
            result = self.engine.process_word("said", include_embedding=True)
        self.assertEqual(result["embedding"], [0.1, 0.2, 0.3])
        fake_model.encode.assert_called_once()

    def test_missing_wordnet_raises_clear_error(self):
        with mock.patch("src.semantic.wn.synsets", side_effect=LookupError("missing")):
            with self.assertRaises(RuntimeError) as context:
                self.engine.semantic_similarity("said", "side")
        self.assertEqual(str(context.exception), WORDNET_SETUP_MESSAGE)


if __name__ == "__main__":
    unittest.main()
