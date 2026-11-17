"""Turn text into vectors with a small local model (fastembed, ONNX on CPU).

Claude does not offer an embeddings endpoint, so embeddings run locally.
That also means uploading documents costs nothing.
"""

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from app import config


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    # Downloaded on first use, then cached.
    return TextEmbedding(model_name=config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[np.ndarray]:
    return list(_model().embed(texts))


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
