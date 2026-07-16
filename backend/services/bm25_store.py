"""
services/bm25_store.py — In-process BM25 keyword index (singleton).

Uses rank-bm25's BM25Okapi implementation. The index is rebuilt from
scratch on every add() call (fast enough at research scale with <10k docs).

Persistence:
    data/bm25.pkl   — pickled (corpus_ids, tokenized_corpus, BM25Okapi)

Usage:
    from services.bm25_store import add, search, corpus_size

    add(["chunk-uuid-1", "chunk-uuid-2"], ["text one", "text two"])
    results = search("some query", k=10)
    # → [("chunk-uuid-1", 3.14), ("chunk-uuid-2", 1.07), …]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_BM25_PATH = _DATA_DIR / "bm25.pkl"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_corpus_ids: list[str] = []       # chunk_id strings (aligned with corpus)
_corpus_texts: list[list[str]] = []  # tokenized texts
_bm25 = None                       # BM25Okapi instance | None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokeniser (lowercase)."""
    return text.lower().split()


def load() -> None:
    """Load BM25 index from disk if it exists. Call from FastAPI startup."""
    global _corpus_ids, _corpus_texts, _bm25
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _BM25_PATH.exists():
        try:
            with _BM25_PATH.open("rb") as f:
                data = pickle.load(f)
            _corpus_ids = data["ids"]
            _corpus_texts = data["texts"]
            _bm25 = data["bm25"]
            log.info("BM25 index loaded from disk (%d documents).", len(_corpus_ids))
        except Exception as exc:
            log.warning("Failed to load BM25 index: %s. Starting fresh.", exc)
            _corpus_ids = []
            _corpus_texts = []
            _bm25 = None
    else:
        log.info("BM25 index initialised (empty).")


def add(chunk_ids: list[str], texts: list[str]) -> None:
    """
    Add new texts to the BM25 index and persist to disk.
    The entire index is rebuilt (fast for research-scale datasets).

    Args:
        chunk_ids: UUID strings identifying each chunk.
        texts:     raw text content (not yet tokenised) for each chunk.
    """
    global _corpus_ids, _corpus_texts, _bm25

    _corpus_ids.extend(chunk_ids)
    _corpus_texts.extend([_tokenize(t) for t in texts])
    _rebuild()
    _save()


def search(query: str, k: int = 50) -> list[tuple[str, float]]:
    """
    Score all documents against the query and return the top-k.

    Returns:
        List of (chunk_id, score) tuples, sorted descending by score.
        Returns [] if the index is empty.
    """
    if _bm25 is None or not _corpus_ids:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        zip(_corpus_ids, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(cid, score) for cid, score in ranked[:k] if score > 0]


def corpus_size() -> int:
    """Return the number of documents in the index."""
    return len(_corpus_ids)


def _rebuild() -> None:
    """Rebuild the BM25Okapi index from the current corpus."""
    global _bm25
    from rank_bm25 import BM25Okapi
    if _corpus_texts:
        _bm25 = BM25Okapi(_corpus_texts)
    else:
        _bm25 = None


def _save() -> None:
    """Persist the corpus and index to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _BM25_PATH.open("wb") as f:
        pickle.dump({"ids": _corpus_ids, "texts": _corpus_texts, "bm25": _bm25}, f)
    log.debug("BM25 index saved (%d documents).", len(_corpus_ids))
