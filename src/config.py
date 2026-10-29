from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model and output directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Semantic embedding model (used only when embeddings are requested)
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# Supported modalities
MODALITIES = [
    "semantic",
    "phonetic",
    "visual"
]
