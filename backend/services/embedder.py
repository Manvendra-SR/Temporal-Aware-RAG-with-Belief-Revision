"""
services/embedder.py — Sentence embedding singleton.

The SentenceTransformer model is loaded once at startup (via `load()`) and
reused across all requests.

Vectors are ALWAYS L2-normalised (``normalize_embeddings=True``). Downstream
code depends on this: `services/retriever.py` converts the squared-L2 distance
returned by FAISS into a cosine similarity using the identity

    ||a - b||² = 2 - 2·cos(a, b)      (valid only for unit vectors)

so normalisation is a load-bearing contract, not an optimisation. It was
previously true only incidentally (the all-MiniLM-L6-v2 pipeline ends with a
Normalize module); requesting it explicitly makes the guarantee independent of
the chosen model.

Usage:
    from services import embedder
    vectors = embedder.embed(["text one", "text two"])  # → np.ndarray (N, 384)
"""

from __future__ import annotations

import logging

import numpy as np

from config import settings

log = logging.getLogger(__name__)

# Lazy global — set by load()
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _resolve_device() -> str:
    """
    Resolve the torch device to load the model on.

    settings.embedding_device is "auto" by default, which selects CUDA when it
    is actually available and CPU otherwise. Pin it to "cpu" or "cuda" via the
    EMBEDDING_DEVICE environment variable to override.
    """
    configured = (settings.embedding_device or "auto").strip().lower()
    if configured != "auto":
        return configured

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken — CPU is always safe
        return "cpu"


def load() -> None:
    """
    Load the SentenceTransformer model into memory.
    Call this once from the FastAPI startup lifespan so the model is
    ready before the first request arrives. Safe to call multiple times.
    """
    global _model
    if _model is not None:
        return
    device = _resolve_device()
    log.info("Loading embedding model '%s' on device '%s' …", MODEL_NAME, device)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME, device=device)
    log.info("Embedding model loaded (dim=%d).", embedding_dim())


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings as L2-normalised vectors.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, each row unit-norm.

    Raises:
        RuntimeError: if the model has not been loaded yet (call load() first).
    """
    if _model is None:
        raise RuntimeError("Embedder not loaded. Call embedder.load() at startup.")
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    vectors = _model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # load-bearing — see module docstring
    )
    return vectors.astype(np.float32)


def embedding_dim() -> int:
    """
    Return the dimensionality of the embedding vectors.

    sentence-transformers renamed `get_sentence_embedding_dimension` to
    `get_embedding_dimension`; both are tried so the app is quiet on new
    versions and still works on older ones.
    """
    if _model is None:
        return EMBEDDING_DIM

    getter = (
        getattr(_model, "get_embedding_dimension", None)
        or getattr(_model, "get_sentence_embedding_dimension", None)
    )
    return getter() if getter else EMBEDDING_DIM
