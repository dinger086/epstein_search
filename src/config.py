"""Configuration settings for the search system."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"

# Database
SQLITE_DB_PATH = DB_DIR / "documents.db"
CHROMA_DB_PATH = DB_DIR / "chroma"

# Ollama models
OCR_MODEL = "glm-ocr:latest"
EMBEDDING_MODEL = "qwen3-embedding:latest"
LLM_MODEL = "qwen3-vl:30b"  # User can override

# Processing settings
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
BATCH_SIZE = 10  # Documents to process at once

# Search settings
TOP_K_RESULTS = 10
HYBRID_KEYWORD_WEIGHT = 0.3
HYBRID_SEMANTIC_WEIGHT = 0.7


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    DB_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
