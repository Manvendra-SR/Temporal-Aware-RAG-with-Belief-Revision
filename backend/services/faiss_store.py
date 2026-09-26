"""
services/faiss_store.py — In-process FAISS vector index (singleton).

Index type: IndexIDMap wrapping IndexFlatIP — an exhaustive (brute-force)
inner-product index, wrapped so that each vector carries its own integer id
(the chunk's faiss_index_id) instead of a positional one.

Why inner product: services/embedder.py L2-normalises every vector it
produces, for both chunks and queries, so the inner product of two of them IS
their cosine similarity. FAISS therefore returns the similarity score directly
and search() can hand it straight to the retriever.

Why flat rather than HNSW: this index used to be IndexHNSWFlat, which searches
an approximate-nearest-neighbour graph and does not implement remove_ids() —
deleting a document could not remove its vectors, so they stayed in the index
as orphans and the only cleanup was a full rebuild. A flat index scans every
vector, which at this project's scale (thousands of chunks, one 384-dim dot
product each) is a fraction of a millisecond, and in exchange the lifecycle
becomes add / search / remove with no special cases. It is also exact, so
search returns the true top-k rather than an approximation.

Persistence:
    data/faiss.index        — the FAISS binary index
    data/faiss_meta.json    — stores the next available integer ID counter

Usage:
    from services.faiss_store import add, remove, search, next_id, index_size

    ids = [next_id(), next_id()]
    add(ids, vectors)                # vectors: np.ndarray (N, 384), unit norm
    ids, sims = search(vec, k=10)    # vec: np.ndarray (384,), unit norm
    remove(ids)                      # exact — the vectors are gone
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
_index = None     # faiss.IndexIDMap wrapping faiss.IndexFlatIP
_next_id: int = 0  # auto-increment counter


def _get_index():
    """Return the FAISS index, creating it if needed."""
    global _index
    if _index is None:
        _index = _build_empty_index()
    return _index


def _build_empty_index():
    import faiss
    # IndexFlatIP: exhaustive inner-product search. Over the unit vectors
    # services/embedder.py produces, inner product == cosine similarity.
    return faiss.IndexIDMap(faiss.IndexFlatIP(EMBEDDING_DIM))


def load() -> None:
    """
    Load index + meta from disk if they exist.
    Call from FastAPI startup lifespan.
    """
    global _index, _next_id
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    if _INDEX_PATH.exists():
        import faiss
        loaded = faiss.read_index(str(_INDEX_PATH))
        if loaded.metric_type == faiss.METRIC_INNER_PRODUCT:
            _index = loaded
            log.info("FAISS index loaded from disk (%d vectors).", _index.ntotal)
        else:
            # An index written by an older build, when this was an L2 HNSW
            # graph. Its scores mean the opposite of what search() now
            # returns, so using it would silently invert every dense ranking.
            # Start empty and say so; rebuild_indexes.py re-embeds from
            # Postgres, which is the source of truth anyway.
            _index = _build_empty_index()
            log.warning(
                "FAISS index at %s uses an old distance metric and was ignored. "
                "Dense retrieval is empty until you run "
                "'python scripts/rebuild_indexes.py'.",
                _INDEX_PATH,
            )
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


def remove(faiss_ids: list[int], persist: bool = True) -> int:
    """
    Delete vectors from the index by their FAISS ids. Returns how many went.

    The ids are the chunks' faiss_index_id values — the same ids add() was
    given, not row ids or positions. Removal is exact and permanent: a flat
    index simply drops the vectors, so nothing is left behind to filter out
    later. The id counter is not rewound, so a removed id is never reused.
    """
    if not faiss_ids:
        return 0

    idx = _get_index()
    removed = int(idx.remove_ids(np.array(faiss_ids, dtype=np.int64)))
    if persist:
        _save()
    log.info("Removed %d vectors from the FAISS index (%d remain).", removed, idx.ntotal)
    return removed


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
    Search for the k most similar vectors to `vector`.

    Args:
        vector: float32 array of shape (EMBEDDING_DIM,), L2-normalised.
        k:      number of results to return.

    Returns:
        (ids, similarities) — lists of length ≤ k, best first. The similarity
        is the inner product, which over unit vectors is cosine similarity in
        [-1, 1]. FAISS pads a short result with id -1; those are dropped.
    """
    idx = _get_index()
    if idx.ntotal == 0:
        return [], []
    query = vector.reshape(1, -1).astype(np.float32)
    k = min(k, idx.ntotal)
    similarities, ids = idx.search(query, k)
    valid = [(int(i), float(s)) for i, s in zip(ids[0], similarities[0]) if i != -1]
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
