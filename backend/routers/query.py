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
from services.belief_revision import RevisionResult, revise as belief_revise
from services.conflict_detector import ConflictResult, detect as detect_conflicts
from services.context import build_context
from services.query_analyzer import QueryAnalysis, analyze as analyze_query
from services.retriever import CandidateChunk, hybrid_retrieve
from services.temporal_reranker import (
    describe_filter,
    describe_time_frame,
    rerank as temporal_rerank,
    scoring_label,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

# Token budget for the assembled LLM context. Candidates beyond this are still
# returned to the client as retrieved sources, but flagged used_in_answer=False.
CONTEXT_TOKEN_BUDGET = 3000

# The temporal filter can remove most of the retrieved candidates (every
# superseded version for a "current" question, everything outside the window
# for a dated one), so the temporal pipeline retrieves this many times
# max_chunks, filters, and only then truncates to max_chunks. The baseline
# retrieves exactly max_chunks, as plain RAG would.
OVERFETCH_FACTOR = 3

# How many top-ranked candidates are scanned for pairwise contradictions.
# Conflict detection is O(n²) in pair selection before the NLI batch, so this
# is capped independently of max_chunks (which may be up to 50).
CONFLICT_SCAN_LIMIT = 20


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_chunks: int = Field(default=20, ge=1, le=50)
    retrieve_only: bool = False
    no_temporal: bool = False  # Phase 8: skip temporal stages for Compare baseline


class SourceResult(BaseModel):
    """
    One retrieved chunk, with every score the pipeline computed for it.

    Score semantics (see services/retriever.py for the full contract):
      bm25_score       raw Okapi BM25 — UNBOUNDED, diagnostic only, never a %
      semantic_score   cosine similarity, [0, 1]
      rrf_score        rank-fusion score, ≤ 2/61 — comparable only within a query
      relevance_score  rrf_score normalised across this query's results, [0, 1]
      temporal_score   freshness weight 2^(-age/half_life), [0, 1]
      composite_score  final reranking score, [0, 1]
    """
    chunk_id: str
    doc_title: str
    snippet: str
    section_heading: Optional[str] = None

    # Retrieval scores
    bm25_score: float
    semantic_score: float
    rrf_score: float
    relevance_score: float

    # Temporal metadata + scores (null when temporal stages were skipped)
    version_string: Optional[str] = None
    published_at: Optional[str] = None
    # End of the chunk's validity window (the date a newer version superseded
    # it); null while the chunk is still current.
    valid_to: Optional[str] = None
    is_latest: Optional[bool] = None
    is_superseded: bool = False
    temporal_score: Optional[float] = None
    composite_score: Optional[float] = None

    # Conflict / belief-revision outcome
    has_conflict: bool = False
    # True when this chunk actually fitted in the LLM's context window. The
    # endpoint returns every candidate, but only those that fit inform the
    # answer — without this the UI cannot honestly label its "Sources" list.
    used_in_answer: bool = False
    # Set when belief revision dropped this chunk, explaining why in plain English.
    excluded_reason: Optional[str] = None
    # 1-based position in the final ranking.
    rank: int = 0


class ConflictInfo(BaseModel):
    """Minimal conflict pair info returned inline with each query response."""
    chunk_id_a: str
    chunk_id_b: str
    conflict_type: str
    nli_score: float


class QueryAnalysisInfo(BaseModel):
    """How the system interpreted the question, surfaced so the UI can show it."""
    # "current" | "point_in_time" | "range" | "version" | "historical" | "atemporal"
    intent: str = "current"
    as_of: Optional[str] = None           # YYYY-MM-DD, point_in_time only
    start_date: Optional[str] = None      # YYYY-MM-DD, range only (inclusive)
    end_date: Optional[str] = None        # YYYY-MM-DD, range only (inclusive)
    version_hint: Optional[str] = None
    temporal_qualifier: bool = False
    wants_historical_sources: bool = False
    # "llm" = interpreted by the LLM; "default" = LLM unavailable or its output
    # failed validation, so the question was treated as "current";
    # "skipped" = baseline mode, where no temporal stage reads the analysis.
    source: str = "default"
    error: Optional[str] = None


def _iso(d) -> Optional[str]:
    return d.isoformat() if d else None


def _analysis_info(analysis: QueryAnalysis, source: str | None = None) -> QueryAnalysisInfo:
    return QueryAnalysisInfo(
        intent=analysis.intent.value,
        as_of=_iso(analysis.as_of),
        start_date=_iso(analysis.start_date),
        end_date=_iso(analysis.end_date),
        version_hint=analysis.version_hint,
        temporal_qualifier=analysis.temporal_qualifier,
        wants_historical_sources=analysis.wants_historical_sources,
        source=source or analysis.source,
        error=analysis.error,
    )


class TemporalFilterInfo(BaseModel):
    """What the temporal filter did, so the UI and evaluation can see it."""
    rule: str                     # e.g. "valid on 2024-03-31"
    scoring: str                  # "relevance + recency" | "relevance only"
    candidates_retrieved: int     # before filtering (over-fetched pool)
    candidates_valid: int         # passed the filter
    candidates_kept: int          # after truncation to max_chunks


class QueryResponse(BaseModel):
    query_id: str
    answer: Optional[str]
    latency_ms: int
    sources: list[SourceResult]
    # How the query was interpreted
    version_hint: Optional[str] = None
    analysis: QueryAnalysisInfo = QueryAnalysisInfo()
    # True when the temporal pipeline ran (false for the baseline comparison)
    temporal_pipeline_applied: bool = True
    # Null in baseline mode
    temporal_filter: Optional[TemporalFilterInfo] = None
    # How many of `sources` actually fitted into the answer's context
    sources_used_in_answer: int = 0
    # Conflict detection results
    conflicts_detected: int = 0
    conflict_pairs: list[ConflictInfo] = []
    # Belief revision output
    answer_confidence: Optional[str] = None      # "high" | "medium" | "low" | "none"
    confidence_reason: Optional[str] = None
    belief_revision_applied: bool = False


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
        0.  Temporal query analysis (LLM → validated intent JSON)
        1.  Hybrid retrieve (BM25 + FAISS + RRF), over-fetching 3× max_chunks
        1b. Temporal filter (validity window for the intent) + rerank,
            then truncate to max_chunks
        1c. Conflict detection (NLI; typed by version lineage)
        1d. Belief revision (conflict arbitration)
        2.  Build context string (token-budgeted, validity-annotated)
        3.  Generate answer via Groq (unless retrieve_only=true)

    With no_temporal=true, steps 0, 1b, 1c and 1d are skipped, max_chunks are
    retrieved, and the context and system prompt carry no version or date
    information: a genuine plain hybrid-RAG baseline over the same retrieval.
        4.  Log to query_log table
        5.  Return answer + source cards
    """
    t_start = time.monotonic()
    query_id = str(uuid.uuid4())
    temporal_enabled = not req.no_temporal

    # ── 0. Temporal query understanding (one LLM call) ─────────────────────
    # Baseline mode runs no temporal stage, so it skips the call entirely —
    # the baseline must not pay for, or be influenced by, temporal analysis.
    if temporal_enabled:
        analysis: QueryAnalysis = analyze_query(req.query)
        analysis_info = _analysis_info(analysis)
    else:
        analysis = QueryAnalysis()
        analysis_info = _analysis_info(analysis, source="skipped")

    # ── 1. Retrieve ──────────────────────────────────────────────────────────
    candidates: list[CandidateChunk] = hybrid_retrieve(
        query=req.query,
        db=db,
        k=req.max_chunks * OVERFETCH_FACTOR if temporal_enabled else req.max_chunks,
    )

    # ── 1b. Temporal filter + rerank, then truncate (skipped in baseline) ────
    temporal_filter: Optional[TemporalFilterInfo] = None
    if temporal_enabled:
        retrieved = len(candidates)
        candidates = temporal_rerank(candidates, analysis)
        valid = len(candidates)
        candidates = candidates[: req.max_chunks]
        temporal_filter = TemporalFilterInfo(
            rule=describe_filter(analysis, datetime.now(timezone.utc).date()),
            scoring=scoring_label(analysis),
            candidates_retrieved=retrieved,
            candidates_valid=valid,
            candidates_kept=len(candidates),
        )

    if not candidates:
        latency_ms = int((time.monotonic() - t_start) * 1000)
        _write_log(db, query_id, req.query, None, latency_ms, [], 0)
        return QueryResponse(
            query_id=query_id,
            answer=None,
            latency_ms=latency_ms,
            sources=[],
            version_hint=analysis.version_hint,
            analysis=analysis_info,
            temporal_pipeline_applied=temporal_enabled,
            temporal_filter=temporal_filter,
        )

    # ── 1c. Conflict detection (skipped in baseline mode) ────────────────────
    conflicts: list[ConflictResult] = (
        detect_conflicts(candidates=candidates[:CONFLICT_SCAN_LIMIT], db=db, query_id=query_id)
        if temporal_enabled else []
    )
    # Build a set of chunk_ids that are part of any conflict
    conflicted_ids: set[str] = set()
    for cr in conflicts:
        conflicted_ids.add(cr.chunk_id_a)
        conflicted_ids.add(cr.chunk_id_b)

    # ── 1d. Belief revision (skipped in baseline mode) ──────────────────────
    # With no conflicts the engine is a pass-through, which is exactly the
    # baseline behaviour, so the baseline path calls it with an empty list
    # rather than branching around it.
    revision: RevisionResult = belief_revise(candidates, conflicts, db, analysis)

    # Excluded chunks never reach the context. (There is no "fall back to all
    # candidates" here: revision always keeps the preferred side of every
    # conflict, so it cannot empty the list, and such a fallback would quietly
    # re-admit exactly the chunks it had excluded.)
    revision_chunk_ids = set(revision.include_chunks)
    context_chunks = [c for c in candidates if c.chunk_id in revision_chunk_ids]

    # ── 2. Build context ─────────────────────────────────────────────────────
    context, used_chunk_ids = build_context(
        context_chunks,
        budget=CONTEXT_TOKEN_BUDGET,
        revision_result=revision if temporal_enabled else None,
        temporal_annotations=temporal_enabled,
        time_frame=describe_time_frame(analysis) if temporal_enabled else None,
    )
    used_in_answer: set[str] = set(used_chunk_ids)

    # ── 3. Generate answer ───────────────────────────────────────────────────
    answer: Optional[str] = None
    if not req.retrieve_only:
        try:
            from services.llm import generate
            answer = generate(context, req.query, temporal=temporal_enabled)
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
            section_heading=c.section_heading,
            rank=i,
            # Retrieval scores
            bm25_score=round(c.bm25_score, 4),
            semantic_score=c.semantic_score,
            rrf_score=c.rrf_score,
            relevance_score=c.relevance_score,
            # Temporal metadata + scores
            version_string=c.version_string,
            published_at=(
                c.published_at.strftime("%Y-%m-%d") if c.published_at else None
            ),
            valid_to=c.valid_to.strftime("%Y-%m-%d") if c.valid_to else None,
            is_latest=c.is_latest,
            is_superseded=c.is_superseded,
            temporal_score=c.temporal_score,
            composite_score=c.composite_score,
            # Conflict / belief-revision outcome
            has_conflict=(c.chunk_id in conflicted_ids),
            used_in_answer=(c.chunk_id in used_in_answer),
            excluded_reason=revision.exclusion_reasons.get(c.chunk_id),
        )
        for i, c in enumerate(candidates, start=1)
    ]

    # ── 5. Log to DB ─────────────────────────────────────────────────────────
    _write_log(db, query_id, req.query, answer, latency_ms, candidates, len(conflicts))

    return QueryResponse(
        query_id=query_id,
        answer=answer,
        latency_ms=latency_ms,
        sources=sources,
        version_hint=analysis.version_hint,
        analysis=analysis_info,
        temporal_pipeline_applied=temporal_enabled,
        temporal_filter=temporal_filter,
        sources_used_in_answer=len(used_in_answer),
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
        # Belief revision fields
        answer_confidence=revision.answer_confidence,
        confidence_reason=revision.confidence_reason,
        belief_revision_applied=revision.belief_revision_applied,
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
