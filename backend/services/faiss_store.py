"""
services/faiss_store.py — In-process FAISS vector index (singleton).

Uses IndexHNSWFlat (Hierarchical Navigable Small World graph) which gives
fast approximate nearest-neighbour search without requiring a GPU.

Persistence:
    data/faiss.index        — the FAISS binary index
    data/faiss_meta.json    — stores the next available integer ID counter

Usage:
    from services.faiss_store import add, search, next_id, index_size

    ids = [next_id(), next_id()]
    add(ids, vectors)              # vectors: np.ndarray (N, 384) float32
    ids, dists = search(vec, k=10) # vec: np.ndarray (384,)
"""

from __future__ import annotations

import json
import logging

import numpy as np

from config import DATA_DIR
from services.embedder import EMBEDDING_DIM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = DATA_DIR
_INDEX_PATH = _DATA_DIR / "faiss.index"
_META_PATH = _DATA_DIR / "faiss_meta.json"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_index = None     # faiss.IndexHNSWFlat
_next_id: int = 0  # auto-increment counter


def _get_index():
    """Return the FAISS index, creating it if needed."""
    global _index
    if _index is None:
        _index = _build_empty_index()
    return _index


def _build_empty_index():
    import faiss
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
    index = faiss.IndexIDMap(index)
    return index


def load() -> None:
    """
    Load index + meta from disk if they exist.
    Call from FastAPI startup lifespan.
    """
    global _index, _next_id
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _INDEX_PATH.exists():
        import faiss
        _index = faiss.read_index(str(_INDEX_PATH))
        log.info("FAISS index loaded from disk (%d vectors).", _index.ntotal)
    else:
        _index = _build_empty_index()
        log.info("FAISS index initialised (empty).")

    if _META_PATH.exists():
        meta = json.loads(_META_PATH.read_text())
        _next_id = meta.get("next_id", 0)
    else:
        _next_id = 0

    log.info("FAISS next_id = %d", _next_id)


def next_id() -> int:
    """Return and increment the global ID counter."""
    global _next_id
    val = _next_id
    _next_id += 1
    return val


def add(faiss_ids: list[int], vectors: np.ndarray, persist: bool = True) -> None:
    """
    Insert vectors into the index.

    Args:
        faiss_ids: integer IDs aligned with vectors rows.
        vectors:   float32 array of shape (N, EMBEDDING_DIM).
        persist:   write the index to disk afterwards. Set False when adding
                   many batches in a row (e.g. a full rebuild) and call
                   persist() once at the end — saving after every batch
                   rewrites the whole index file each time.
    """
    idx = _get_index()
    ids_arr = np.array(faiss_ids, dtype=np.int64)
    idx.add_with_ids(vectors.astype(np.float32), ids_arr)
    if persist:
        _save()


def reset() -> None:
    """
    Drop all vectors and restart the ID counter, without touching disk.

    Used by scripts/rebuild_indexes.py to rebuild from the database. Call
    persist() afterwards to write the result out.
    """
    global _index, _next_id
    _index = _build_empty_index()
    _next_id = 0
    log.info("FAISS index reset (empty, next_id=0).")


def persist() -> None:
    """Write the current index and ID counter to disk."""
    _save()


def search(vector: np.ndarray, k: int = 10) -> tuple[list[int], list[float]]:
    """
    Search for the k nearest neighbours of `vector`.

    Args:
        vector: float32 array of shape (EMBEDDING_DIM,).
        k:      number of results to return.

    Returns:
        (ids, distances) — lists of length ≤ k.
    """
    idx = _get_index()
    if idx.ntotal == 0:
        return [], []
    query = vector.reshape(1, -1).astype(np.float32)
    k = min(k, idx.ntotal)
    distances, ids = idx.search(query, k)
    valid = [(int(i), float(d)) for i, d in zip(ids[0], distances[0]) if i != -1]
    return [x[0] for x in valid], [x[1] for x in valid]


def index_size() -> int:
    """Return the number of vectors currently in the index."""
    if _index is None:
        return 0
    return _index.ntotal


def _save() -> None:
    """Persist the index and metadata counter to disk."""
    import faiss
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(_get_index(), str(_INDEX_PATH))
    _META_PATH.write_text(json.dumps({"next_id": _next_id}))
    log.debug("FAISS index saved (%d vectors, next_id=%d).", _get_index().ntotal, _next_id)
