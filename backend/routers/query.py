"""
routers/query.py — Query endpoint.

Routes:
    POST /api/v1/query
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from models import QueryLog
from services.conflict_detector import ConflictResult, detect as detect_conflicts
from services.context import build_context
from services.query_analyzer import QueryAnalysis, analyze as analyze_query
from services.retriever import CandidateChunk, hybrid_retrieve
from services.temporal_reranker import rerank as temporal_rerank

log = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_chunks: int = Field(default=20, ge=1, le=50)
    retrieve_only: bool = False


class SourceResult(BaseModel):
    """
    Full source schema — future-phase fields are pre-declared as Optional
    so no breaking API change is needed in Phases 5 and 6.
    """
    chunk_id: str
    doc_title: str
    snippet: str
    bm25_score: float
    semantic_score: float
    rrf_score: float
    # Phase 5+ (null until temporal scoring is implemented):
    version_string: Optional[str] = None
    published_at: Optional[str] = None
    is_latest: Optional[bool] = None
    temporal_score: Optional[float] = None
    composite_score: Optional[float] = None
    # Phase 6+ (null until conflict detection is implemented):
    has_conflict: Optional[bool] = None


class ConflictInfo(BaseModel):
    """Minimal conflict pair info returned inline with each query response."""
    chunk_id_a: str
    chunk_id_b: str
    conflict_type: str
    nli_score: float


class QueryResponse(BaseModel):
    query_id: str
    answer: Optional[str]
    latency_ms: int
    sources: list[SourceResult]
    # Phase 5+: version hint extracted from query (null if none detected)
    version_hint: Optional[str] = None
    # Phase 6+: conflict detection results
    conflicts_detected: int = 0
    conflict_pairs: list[ConflictInfo] = []


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query_endpoint(
    req: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """
    Full RAG pipeline:
        1. Hybrid retrieve (BM25 + FAISS + RRF)
        1b. Temporal rerank
        1c. Conflict detection
        2. Build context string (token-budgeted)
        3. Generate answer via Groq (unless retrieve_only=true)
        4. Log to query_log table
        5. Return answer + source cards
    """
    t_start = time.monotonic()
    query_id = str(uuid.uuid4())

    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    candidates: list[CandidateChunk] = hybrid_retrieve(
        query=req.query,
        db=db,
        k=req.max_chunks,
    )

    # ── 1b. Temporal rerank ───────────────────────────────────────────
    analysis: QueryAnalysis = analyze_query(req.query)
    if candidates:
        candidates = temporal_rerank(candidates, analysis, db)

    if not candidates:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        _write_log(db, query_id, req.query, None, latency_ms, [], 0)
        return QueryResponse(
            query_id=query_id,
            answer=None,
            latency_ms=latency_ms,
            sources=[],
            version_hint=analysis.version_hint,
        )

    # ── 1c. Conflict detection ────────────────────────────────────────────────
    conflicts: list[ConflictResult] = detect_conflicts(
        candidates=candidates[:20],
        db=db,
        query_id=query_id,
    )
    # Build a set of chunk_ids that are part of any conflict
    conflicted_ids: set[str] = set()
    for cr in conflicts:
        conflicted_ids.add(cr.chunk_id_a)
        conflicted_ids.add(cr.chunk_id_b)

    # ── 2. Build context ─────────────────────────────────────────────────────
    context, _ = build_context(candidates, budget=3000)

    # ── 3. Generate answer ───────────────────────────────────────────────────
    answer: Optional[str] = None
    if not req.retrieve_only:
        try:
            from services.llm import generate
            answer = generate(context, req.query)
        except RuntimeError as exc:
            # No API key set — surface a helpful message instead of 500
            log.warning("LLM call skipped: %s", exc)
            answer = f"[LLM unavailable: {exc}]"
        except Exception as exc:
            log.error("LLM call failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    latency_ms = int((time.monotonic() - t_start) * 1000)

    # ── 4. Build source cards ─────────────────────────────────────────────────
    sources = [
        SourceResult(
            chunk_id=c.chunk_id,
            doc_title=c.doc_title,
            snippet=c.content_snippet,
            bm25_score=round(c.bm25_score, 4),
            semantic_score=c.semantic_score,
            rrf_score=c.rrf_score,
            # Phase 5 temporal fields
            version_string=c.version_string,
            published_at=(
                c.published_at.strftime("%Y-%m-%d") if c.published_at else None
            ),
            is_latest=c.is_latest,
            temporal_score=c.temporal_score,
            composite_score=c.composite_score,
            # Phase 6: flag chunks that appear in any conflict pair
            has_conflict=(c.chunk_id in conflicted_ids),
        )
        for c in candidates
    ]

    # ── 5. Log to DB ─────────────────────────────────────────────────────────
    _write_log(db, query_id, req.query, answer, latency_ms, candidates, len(conflicts))

    return QueryResponse(
        query_id=query_id,
        answer=answer,
        latency_ms=latency_ms,
        sources=sources,
        version_hint=analysis.version_hint,
        conflicts_detected=len(conflicts),
        conflict_pairs=[
            ConflictInfo(
                chunk_id_a=cr.chunk_id_a,
                chunk_id_b=cr.chunk_id_b,
                conflict_type=cr.conflict_type,
                nli_score=cr.nli_score,
            )
            for cr in conflicts
        ],
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_log(
    db: Session,
    query_id: str,
    query_text: str,
    answer: Optional[str],
    latency_ms: int,
    candidates: list[CandidateChunk],
    conflicts_detected: int = 0,
) -> None:
    """Write a row to the query_log table."""
    try:
        chunk_ids = [c.chunk_id for c in candidates]
        db.add(QueryLog(
            query_id=query_id,
            query_text=query_text,
            queried_at=datetime.now(timezone.utc),
            retrieved_chunk_ids=chunk_ids,
            answer_text=answer,
            latency_ms=latency_ms,
            conflicts_detected=conflicts_detected,
        ))
        db.commit()
    except Exception as exc:
        log.warning("Failed to write query_log: %s", exc)
        db.rollback()
