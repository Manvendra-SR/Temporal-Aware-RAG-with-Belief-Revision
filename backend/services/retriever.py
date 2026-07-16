"""
services/retriever.py — Hybrid BM25 + Dense retrieval with RRF fusion.

hybrid_retrieve(query, db, k, domain_filter) → List[CandidateChunk]

Pipeline:
  1. BM25 keyword search  → top-50 chunk_ids + scores
  2. Dense vector search  → top-50 faiss_ids + L2 distances
  3. RRF merge            → combined score per chunk
  4. Fetch DB metadata    → CandidateChunk dataclass
  5. Optional domain filter applied post-fetch
  6. Return top-k sorted by rrf_score descending
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
    domain: str
    content: str
    content_snippet: str
    section_heading: str | None
    bm25_score: float
    semantic_score: float   # converted from L2 distance: 1 / (1 + distance)
    rrf_score: float


def hybrid_retrieve(
    query: str,
    db: Session,
    k: int = 20,
    domain_filter: str | None = None,
) -> list[CandidateChunk]:
    """
    Retrieve the top-k most relevant chunks for `query` using hybrid search.

    Args:
        query:         Natural language query string.
        db:            Active SQLAlchemy session.
        k:             Number of results to return.
        domain_filter: If set, only return chunks from this domain.

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

        # Skip if domain filter is active
        if domain_filter and doc_row.domain != domain_filter:
            continue

        dist = dense_distance_map.get(cid, 1e6)
        semantic = 1.0 / (1.0 + dist)    # convert L2 distance to similarity

        results.append(CandidateChunk(
            chunk_id=cid,
            faiss_id=chunk_row.faiss_index_id or -1,
            doc_id=doc_row.doc_id,
            doc_title=doc_row.title,
            domain=doc_row.domain,
            content=chunk_row.content,
            content_snippet=chunk_row.content_snippet or chunk_row.content[:200],
            section_heading=chunk_row.section_heading,
            bm25_score=bm25_score_map.get(cid, 0.0),
            semantic_score=round(semantic, 4),
            rrf_score=round(rrf_scores[cid], 6),
        ))

    # Sort final list by rrf_score descending, take top-k
    results.sort(key=lambda c: c.rrf_score, reverse=True)
    log.info(
        "hybrid_retrieve: query=%r → %d candidates, returning %d",
        query[:60], len(top_candidates), min(k, len(results))
    )
    return results[:k]
