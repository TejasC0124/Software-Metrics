from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image


class VisualEngine:
    """Create current orthographic and handwriting representations.

    Text responses use spelling similarity today. Handwriting images can be
    validated and prepared for a future trained visual encoder.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, image_size: tuple[int, int] = (128, 128)):
        self.image_size = image_size

    @staticmethod
    def spelling_similarity(target: str, response: str) -> float:
        return SequenceMatcher(
            None, str(target).strip().lower(), str(response).strip().lower()
        ).ratio()

    def is_valid_image(self, image_path: str | Path) -> bool:
        image_path = Path(image_path)
        if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except (OSError, ValueError):
            return False

    def load_image(self, image_path: str | Path) -> Image.Image | None:
        image_path = Path(image_path)
        if not self.is_valid_image(image_path):
            return None
        try:
            with Image.open(image_path) as image:
                return image.convert("L").resize(self.image_size)
        except (OSError, ValueError):
            return None

    def process_image(self, image_path: str | Path) -> dict[str, object]:
        image = self.load_image(image_path)
        return {
            "image_path": str(image_path),
            "valid": image is not None,
            "image": image,
        }
