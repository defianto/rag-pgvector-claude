"""App settings, read from environment variables (see .env.example)."""

import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"
)

# Claude
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # low | medium | high
CLAUDE_MAX_TOKENS = _int("CLAUDE_MAX_TOKENS", 4000)

# Embeddings (runs locally, no API key needed). Multilingual so it works
# for English and Indonesian PDFs. Output size must match db/init.sql.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM = 384

# Chunking and retrieval
CHUNK_SIZE = _int("CHUNK_SIZE", 1000)  # characters
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)
TOP_K = _int("TOP_K", 5)
