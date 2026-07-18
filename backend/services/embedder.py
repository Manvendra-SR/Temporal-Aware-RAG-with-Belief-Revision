"""
services/embedder.py — Sentence embedding singleton.

The SentenceTransformer model is loaded once at module import time
and reused across all requests. Call `load()` from the FastAPI lifespan
to trigger the download before the first request arrives.

Usage:
    from services.embedder import embed
    vectors = embed(["text one", "text two"])  # → np.ndarray (N, 384)
"""

from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

# Lazy global — set by load()
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def load() -> None:
    """
    Load the SentenceTransformer model into memory.
    Call this once from the FastAPI startup lifespan so the model is
    ready before the first request arrives. Safe to call multiple times.
    """
    global _model
    if _model is not None:
        return
    log.info("Loading embedding model '%s' …", MODEL_NAME)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME, device="cuda")
    log.info("Embedding model loaded (dim=%d).", _model.get_sentence_embedding_dimension())


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32.

    Raises:
        RuntimeError: if the model has not been loaded yet (call load() first).
    """
    if _model is None:
        raise RuntimeError("Embedder not loaded. Call embedder.load() at startup.")
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    vectors = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def embedding_dim() -> int:
    """Return the dimensionality of the embedding vectors."""
    if _model is None:
        return 384  # default for all-MiniLM-L6-v2
    return _model.get_sentence_embedding_dimension()
