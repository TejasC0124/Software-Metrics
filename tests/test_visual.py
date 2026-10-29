import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.visual import VisualEngine


class VisualEngineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.engine = VisualEngine()

    def tearDown(self):
        self._tmp.cleanup()

    def test_spelling_similarity_is_case_insensitive(self):
        self.assertEqual(VisualEngine.spelling_similarity("Said", "said"), 1.0)
        self.assertAlmostEqual(VisualEngine.spelling_similarity("said", "side"), 0.75)

    def test_valid_png_is_accepted_and_resized_to_greyscale(self):
        path = self.dir / "word.png"
        Image.new("RGB", (300, 60), "white").save(path)
        self.assertTrue(self.engine.is_valid_image(path))
        image = self.engine.load_image(path)
        self.assertEqual(image.size, (128, 128))
        self.assertEqual(image.mode, "L")

    def test_text_file_with_image_extension_is_rejected(self):
        path = self.dir / "fake.png"
        path.write_text("not an image", encoding="utf-8")
        self.assertFalse(self.engine.is_valid_image(path))
        self.assertIsNone(self.engine.load_image(path))

    def test_unsupported_extension_is_rejected(self):
        path = self.dir / "word.gif"
        Image.new("L", (10, 10)).save(path)
        self.assertFalse(self.engine.is_valid_image(path))


if __name__ == "__main__":
    unittest.main()
