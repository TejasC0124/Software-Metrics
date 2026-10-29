import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.visual_preprocess import validate_image


class VisualPreprocessTestCase(unittest.TestCase):
    def test_valid_and_corrupt_images_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.png"
            bad = Path(directory) / "bad.png"
            Image.new("L", (40, 20), 255).save(good)
            bad.write_bytes(b"corrupt")
            self.assertEqual(validate_image(good), (True, 40, 20, "L"))
            self.assertEqual(validate_image(bad), (False, None, None, None))


if __name__ == "__main__":
    unittest.main()
