from pathlib import Path
import cmudict
from difflib import SequenceMatcher


class PhoneticEngine:
    """
    SPBE Phonetic Representation Engine.

    Uses CMUdict to generate phonetic representations
    and calculate phonetic similarity.
    """

    def __init__(self):
        self.dictionary = cmudict.dict()

    def get_pronunciations(self, word):
        """Return all available pronunciations for a word."""
        return self.dictionary.get(word.lower(), [])

    def get_phonemes(self, word):
        """Return the primary pronunciation."""
        pronunciations = self.get_pronunciations(word)

        if not pronunciations:
            return None

        return pronunciations[0]

    def phonetic_similarity(self, word1, word2):
        """Calculate phonetic similarity between two words."""

        phonemes1 = self.get_phonemes(word1)
        phonemes2 = self.get_phonemes(word2)

        if phonemes1 is None or phonemes2 is None:
            return None

        return SequenceMatcher(
            None,
            phonemes1,
            phonemes2
        ).ratio()

    def process_word(self, word):
        """Create complete phonetic representation."""

        pronunciations = self.get_pronunciations(word)

        return {
            "word": word,
            "cmudict_available": len(pronunciations) > 0,
            "num_pronunciations": len(pronunciations),
            "pronunciations": pronunciations,
            "primary_phonemes": (
                pronunciations[0]
                if pronunciations
                else None
            )
        }
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model and output directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Semantic model
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# Supported modalities
MODALITIES = [
    "semantic",
    "phonetic",
    "visual"
]