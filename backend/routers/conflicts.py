"""
routers/conflicts.py — Conflict records API.

Routes:
    GET /api/v1/conflicts
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models import Chunk, ConflictPair, Document

log = logging.getLogger(__name__)

router = APIRouter(tags=["conflicts"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ChunkInfo(BaseModel):
    chunk_id: str
    snippet: str
    section_heading: Optional[str] = None
    doc_title: str
    version_string: Optional[str] = None
    published_at: Optional[str] = None
    is_latest: bool


class ConflictRecord(BaseModel):
    conflict_id: str
    conflict_type: str
    nli_score: float
    detected_at: str
    detected_during: Optional[str] = None   # query_id
    is_resolved: bool
    resolution_type: Optional[str] = None
    chunk_a: ChunkInfo
    chunk_b: ChunkInfo


class ConflictsResponse(BaseModel):
    total: int          # matching the current filters
    page: int
    limit: int
    conflicts: list[ConflictRecord]
    # Corpus-wide counts, independent of filters and paging. The UI previously
    # derived "pending"/"resolved" by counting the current page while showing
    # `total` next to them, so the three numbers disagreed as soon as there was
    # more than one page.
    total_unresolved: int = 0
    total_resolved: int = 0


class ResolveRequest(BaseModel):
    resolution_type: str   # "temporal_preference" | "manual" | "scope_clarification"
    resolution_note: str = ""


# ---------------------------------------------------------------------------
# GET /api/v1/conflicts
# ---------------------------------------------------------------------------


@router.get("/conflicts", response_model=ConflictsResponse)
def list_conflicts(
    is_resolved: Optional[bool] = Query(default=None, description="Filter by resolution status"),
    conflict_type: Optional[str] = Query(default=None, alias="type", description="Filter by conflict type"),
    detected_during: Optional[str] = Query(default=None, description="Filter by query_id"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ConflictsResponse:
    """
    List detected conflict pairs with both chunk snippets.

    Supports filtering by resolution status, type, and originating query.
    Results are paginated (default 20 per page).
    """
    q = db.query(ConflictPair)

    if is_resolved is not None:
        q = q.filter(ConflictPair.is_resolved == is_resolved)
    if conflict_type:
        q = q.filter(ConflictPair.conflict_type == conflict_type)
    if detected_during:
        q = q.filter(ConflictPair.detected_during == detected_during)

    total = q.count()
    offset = (page - 1) * limit
    rows: list[ConflictPair] = (
        q.order_by(ConflictPair.detected_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    records: list[ConflictRecord] = []
    for row in rows:
        chunk_a_info = _fetch_chunk_info(db, row.chunk_id_a)
        chunk_b_info = _fetch_chunk_info(db, row.chunk_id_b)
        if chunk_a_info is None or chunk_b_info is None:
            # Orphaned conflict — skip (chunk was deleted)
            continue
        records.append(ConflictRecord(
            conflict_id=row.conflict_id,
            conflict_type=row.conflict_type or "unknown",
            nli_score=round(row.nli_score or 0.0, 4),
            detected_at=row.detected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            detected_during=row.detected_during,
            is_resolved=row.is_resolved,
            resolution_type=row.resolution_type,
            chunk_a=chunk_a_info,
            chunk_b=chunk_b_info,
        ))

    total_unresolved = (
        db.query(func.count(ConflictPair.conflict_id))
        .filter(ConflictPair.is_resolved.is_(False))
        .scalar() or 0
    )
    total_resolved = (
        db.query(func.count(ConflictPair.conflict_id))
        .filter(ConflictPair.is_resolved.is_(True))
        .scalar() or 0
    )

    return ConflictsResponse(
        total=total,
        page=page,
        limit=limit,
        conflicts=records,
        total_unresolved=total_unresolved,
        total_resolved=total_resolved,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/conflicts/{conflict_id}/resolve
# ---------------------------------------------------------------------------


@router.post("/conflicts/{conflict_id}/resolve", response_model=ConflictRecord)
def resolve_conflict(
    conflict_id: str,
    body: ResolveRequest,
    db: Session = Depends(get_db),
) -> ConflictRecord:
    """
    Mark a conflict as resolved with a resolution type and optional note.

    Future queries will use this stored resolution instead of re-running NLI,
    and belief_revision.py will apply the stored resolution_type directly.
    """
    row: ConflictPair | None = (
        db.query(ConflictPair)
        .filter(ConflictPair.conflict_id == conflict_id)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Conflict {conflict_id!r} not found.")

    if row.is_resolved:
        raise HTTPException(
            status_code=400,
            detail=f"Conflict {conflict_id!r} is already resolved.",
        )

    # Apply resolution
    row.is_resolved = True
    row.resolution_type = body.resolution_type
    row.resolution_note = body.resolution_note or None
    row.resolved_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(row)
    except Exception as exc:
        db.rollback()
        log.error("Failed to resolve conflict %s: %s", conflict_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while resolving conflict.")

    # Return the full updated record
    chunk_a_info = _fetch_chunk_info(db, row.chunk_id_a)
    chunk_b_info = _fetch_chunk_info(db, row.chunk_id_b)
    if chunk_a_info is None or chunk_b_info is None:
        raise HTTPException(status_code=500, detail="Chunk data unavailable for this conflict.")

    return ConflictRecord(
        conflict_id=row.conflict_id,
        conflict_type=row.conflict_type or "unknown",
        nli_score=round(row.nli_score or 0.0, 4),
        detected_at=row.detected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        detected_during=row.detected_during,
        is_resolved=row.is_resolved,
        resolution_type=row.resolution_type,
        chunk_a=chunk_a_info,
        chunk_b=chunk_b_info,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fetch_chunk_info(db: Session, chunk_id: str) -> ChunkInfo | None:
    """Load a chunk + its parent document and return a ChunkInfo."""
    row = (
        db.query(Chunk, Document)
        .join(Document, Chunk.doc_id == Document.doc_id)
        .filter(Chunk.chunk_id == chunk_id)
        .first()
    )
    if row is None:
        return None
    chunk, doc = row
    return ChunkInfo(
        chunk_id=chunk.chunk_id,
        snippet=chunk.content_snippet or chunk.content[:250],
        section_heading=chunk.section_heading,
        doc_title=doc.title,
        version_string=doc.version_string,
        published_at=(doc.published_at.strftime("%Y-%m-%d") if doc.published_at else None),
        is_latest=bool(doc.is_latest),
    )
