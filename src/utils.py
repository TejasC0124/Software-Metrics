from pathlib import Path


def ensure_directory(path):
    """
    Create a directory if it does not already exist.
    """

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def normalize_word(word):
    """
    Basic word normalization.
    """

    if word is None:
        return ""

    return str(word).strip().lower()