"""
services/retriever.py — Hybrid BM25 + Dense retrieval with RRF fusion.

hybrid_retrieve(query, db, k) → List[CandidateChunk]

Pipeline:
  1. BM25 keyword search  → top-50 chunk_ids + raw BM25 scores
  2. Dense vector search  → top-50 faiss_ids + squared-L2 distances
  3. RRF merge            → combined rank-fusion score per chunk
  4. Fetch DB metadata    → CandidateChunk dataclass (incl. temporal fields)
  5. Normalise RRF into `relevance_score` ∈ [0, 1] across the candidate set
  6. Return top-k sorted by rrf_score descending
  (temporal_reranker then re-sorts these before query.py returns them)

Score semantics — read this before displaying or combining any of these
-----------------------------------------------------------------------
bm25_score       Raw Okapi BM25. UNBOUNDED and corpus-dependent; typical values
                 range from 0 to ~30. It is NOT a probability or a percentage.
                 Diagnostic only — never combine or render it as a fraction.

semantic_score   True cosine similarity in [0, 1]. Embeddings are unit vectors
                 (see services/embedder.py), and FAISS returns SQUARED L2, so
                 the exact conversion is  cos = 1 - d/2.
                 This replaced `1 / (1 + d)`, which was monotonic but crushed
                 every result into a ~0.40-0.43 band, making the term
                 contribute almost nothing to any weighted combination.

rrf_score        Reciprocal Rank Fusion, sum of 1/(60 + rank) over the two
                 retrievers. Bounded by 2/61 ≈ 0.0328. Rank-based, so it is
                 comparable WITHIN one query's results but meaningless across
                 queries and definitely not a percentage.

relevance_score  rrf_score min-max normalised across this query's candidates,
                 so the best candidate is 1.0 and the worst is 0.0. This is the
                 term the temporal reranker combines with temporal decay, and
                 it is the only relevance number safe to render as a bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from models import Chunk, Document
from services import bm25_store, embedder, faiss_store

log = logging.getLogger(__name__)

RRF_K = 60          # constant in the RRF formula — standard value
BM25_FETCH = 50     # how many BM25 candidates to pull
DENSE_FETCH = 50    # how many dense candidates to pull


@dataclass
class CandidateChunk:
    chunk_id: str
    faiss_id: int
    doc_id: str
    doc_title: str
    content: str
    content_snippet: str
    section_heading: str | None
    bm25_score: float       # raw Okapi BM25, unbounded — diagnostic only
    semantic_score: float   # cosine similarity in [0, 1]
    rrf_score: float        # reciprocal rank fusion, ≤ 2/61
    relevance_score: float = 0.0   # rrf_score normalised to [0, 1] per query

    # ── Temporal metadata (populated from the DB join) ───────────────────────
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_superseded: bool = False
    version_string: str | None = None
    published_at: datetime | None = None
    is_latest: bool = True

    # ── Scores computed by temporal_reranker (None until reranked) ───────────
    temporal_score: float | None = None
    version_boost: float | None = None
    latest_bonus: float | None = None
    composite_score: float | None = None


def hybrid_retrieve(
    query: str,
    db: Session,
    k: int = 20,
) -> list[CandidateChunk]:
    """
    Retrieve the top-k most relevant chunks for `query` using hybrid search.

    Args:
        query:  Natural language query string.
        db:     Active SQLAlchemy session.
        k:      Number of results to return.

    Returns:
        List of CandidateChunk sorted descending by rrf_score.
    """
    # ── 1. BM25 search ──────────────────────────────────────────────────────
    bm25_results: list[tuple[str, float]] = bm25_store.search(query, k=BM25_FETCH)
    # {chunk_id: (bm25_rank, bm25_score)}
    bm25_rank: dict[str, int] = {cid: rank for rank, (cid, _) in enumerate(bm25_results)}
    bm25_score_map: dict[str, float] = {cid: score for cid, score in bm25_results}

    # ── 2. Dense search ─────────────────────────────────────────────────────
    query_vec = embedder.embed([query])[0]  # shape (384,)
    dense_ids, dense_dists = faiss_store.search(query_vec, k=DENSE_FETCH)
    # Convert FAISS integer IDs → chunk_ids via DB
    faiss_id_to_chunk_id: dict[int, str] = {}
    if dense_ids:
        db_chunks = (
            db.query(Chunk.chunk_id, Chunk.faiss_index_id)
            .filter(Chunk.faiss_index_id.in_(dense_ids))
            .all()
        )
        faiss_id_to_chunk_id = {row.faiss_index_id: row.chunk_id for row in db_chunks}

    dense_rank: dict[str, int] = {}
    dense_distance_map: dict[str, float] = {}
    for rank, (fid, dist) in enumerate(zip(dense_ids, dense_dists)):
        cid = faiss_id_to_chunk_id.get(fid)
        if cid:
            dense_rank[cid] = rank
            dense_distance_map[cid] = float(dist)

    # ── 3. RRF merge ────────────────────────────────────────────────────────
    all_chunk_ids = set(bm25_rank) | set(dense_rank)
    rrf_scores: dict[str, float] = {}
    for cid in all_chunk_ids:
        r_bm25 = bm25_rank.get(cid, BM25_FETCH)    # missing → worst rank
        r_dense = dense_rank.get(cid, DENSE_FETCH)
        rrf_scores[cid] = 1.0 / (RRF_K + r_bm25) + 1.0 / (RRF_K + r_dense)

    # Sort by RRF score descending, keep top candidates for DB fetch
    top_candidates = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:k * 2]

    if not top_candidates:
        log.info("hybrid_retrieve: no candidates found for query=%r", query[:60])
        return []

    # ── 4. Fetch DB metadata ─────────────────────────────────────────────────
    rows = (
        db.query(Chunk, Document)
        .join(Document, Chunk.doc_id == Document.doc_id)
        .filter(Chunk.chunk_id.in_(top_candidates))
        .all()
    )

    results: list[CandidateChunk] = []
    for chunk_row, doc_row in rows:
        cid = chunk_row.chunk_id

        results.append(CandidateChunk(
            chunk_id=cid,
            faiss_id=chunk_row.faiss_index_id or -1,
            doc_id=doc_row.doc_id,
            doc_title=doc_row.title,
            content=chunk_row.content,
            content_snippet=chunk_row.content_snippet or chunk_row.content[:200],
            section_heading=chunk_row.section_heading,
            bm25_score=bm25_score_map.get(cid, 0.0),
            semantic_score=cosine_from_sq_l2(dense_distance_map.get(cid)),
            rrf_score=round(rrf_scores[cid], 6),
            # ── Temporal metadata ─────────────────────────────────────────
            valid_from=chunk_row.valid_from,
            valid_to=chunk_row.valid_to,
            is_superseded=bool(chunk_row.is_superseded),
            version_string=doc_row.version_string,
            published_at=doc_row.published_at,
            is_latest=bool(doc_row.is_latest),
        ))

    # Sort final list by rrf_score descending, take top-k
    results.sort(key=lambda c: c.rrf_score, reverse=True)
    results = results[:k]

    # Normalise fused relevance across the returned set so the reranker has a
    # relevance term that actually spans [0, 1]. Done after truncation so the
    # scale reflects the candidates the user is shown.
    _assign_relevance_scores(results)

    # The DB fetch can return fewer rows than requested when the FAISS/BM25
    # indexes contain entries whose chunks no longer exist in Postgres. Surface
    # that drift instead of silently returning a short list.
    missing = len(top_candidates) - len(rows)
    if missing > 0:
        log.warning(
            "hybrid_retrieve: %d/%d fused candidates had no matching chunk row — "
            "search indexes are out of sync with the database. "
            "Run `python scripts/rebuild_indexes.py` to rebuild them.",
            missing, len(top_candidates),
        )

    log.info(
        "hybrid_retrieve: query=%r → %d fused candidates, returning %d",
        query[:60], len(top_candidates), len(results),
    )
    return results


def cosine_from_sq_l2(sq_distance: float | None) -> float:
    """
    Convert a FAISS squared-L2 distance between UNIT vectors to cosine similarity.

    For unit vectors, ||a - b||² = 2 - 2·cos(a, b), hence cos = 1 - d/2.
    Embeddings are L2-normalised by services.embedder, so this identity holds.

    A chunk that BM25 surfaced but the dense retriever never returned has no
    distance; it scores 0.0 rather than an arbitrary large-distance value.
    Result is clamped to [0, 1] — negative cosines mean "unrelated" here.
    """
    if sq_distance is None:
        return 0.0
    cosine = 1.0 - (float(sq_distance) / 2.0)
    return round(max(0.0, min(1.0, cosine)), 4)


def _assign_relevance_scores(candidates: list[CandidateChunk]) -> None:
    """
    Min-max normalise rrf_score across `candidates` into `relevance_score`.

    RRF values are tiny (≤ 2/61) and clustered, so they are unusable as a
    weighted term or as a UI bar in raw form. Normalising per query gives a
    well-spread [0, 1] relevance signal. Mutates the candidates in place.

    With a single candidate, or when every candidate ties, everything scores
    1.0 — there is no meaningful spread to express.
    """
    if not candidates:
        return

    scores = [c.rrf_score for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo

    for cand in candidates:
        cand.relevance_score = (
            1.0 if span <= 0 else round((cand.rrf_score - lo) / span, 4)
        )
