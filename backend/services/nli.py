"""
services/nli.py — NLI cross-encoder singleton.

Wraps CrossEncoder("cross-encoder/nli-deberta-v3-small") as a module-level
singleton so the model is loaded exactly once (at startup) and shared across
all requests.

Usage
-----
    from services import nli
    scores = nli.predict([("text A", "text B"), ...])
    # Returns list[float]: P(contradiction) for each pair.

Startup
-------
    Call nli.load() in the FastAPI lifespan startup event.
    After load(), predict() is safe to call from any request handler.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# ── Singleton ────────────────────────────────────────────────────────────────

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# The CrossEncoder instance — None until load() is called.
_model: Optional[object] = None   # type: sentence_transformers.CrossEncoder


def load() -> None:
    """
    Load the NLI cross-encoder model into memory.

    Downloads ~80 MB on first call; subsequent calls return immediately from
    the Hugging Face cache.  Call this once from the FastAPI startup event.
    """
    global _model
    if _model is not None:
        log.debug("nli.load(): model already loaded, skipping.")
        return

    log.info("Loading NLI model '%s' …", _MODEL_NAME)
    t0 = time.monotonic()
    try:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(_MODEL_NAME)
        elapsed = (time.monotonic() - t0) * 1000
        log.info("NLI model loaded in %.0f ms.", elapsed)
    except Exception as exc:
        log.error("Failed to load NLI model: %s", exc, exc_info=True)
        raise


def predict(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Compute P(contradiction) for each (text_a, text_b) pair.

    Args:
        pairs: List of (premise, hypothesis) text pairs.

    Returns:
        List of floats in [0, 1] — higher means more likely contradictory.

    Raises:
        RuntimeError: If load() has not been called yet.
    """
    if _model is None:
        raise RuntimeError(
            "NLI model is not loaded. Call nli.load() at startup before predict()."
        )
    if not pairs:
        return []

    # CrossEncoder returns a 2-D array: [N, 3] for NLI (entail / neutral / contra)
    # The contradiction label index is 0 for deberta-v3-small trained on NLI.
    # We use softmax to get probabilities.
    import numpy as np

    raw = _model.predict(pairs, apply_softmax=True)   # shape (N, 3)
    # Label order for cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
    # Index 0 = contradiction
    contradiction_scores: list[float] = [float(row[0]) for row in raw]
    return contradiction_scores


def is_loaded() -> bool:
    """Return True if the model has been successfully loaded."""
    return _model is not None
