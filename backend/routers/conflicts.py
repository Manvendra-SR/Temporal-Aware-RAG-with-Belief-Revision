"""
routers/conflicts.py — Conflict records and human review.

Routes:
    GET  /api/v1/conflicts                     list, filtered by review status
    POST /api/v1/conflicts/{conflict_id}/resolve   record (or change) a decision

Review model
------------
Only a direct_contradiction (two unrelated documents disagree) is reviewed.
A version_supersession is a document changing between its own versions: its
lineage already explains it, and no decision about it could change an answer
(a lineage's validity windows never overlap), so it is listed for information
and cannot be resolved.

A decision is one of:
    prefer_source  + preferred_chunk_id   one side is right
    both_valid                            not really a contradiction
"Not sure yet" is not a decision: the conflict simply stays pending, and the
automatic rules in belief_revision.py apply. A decision can be changed by
resolving the conflict again.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models import (
    BOTH_VALID,
    CONTRADICTION,
    PREFER_SOURCE,
    VERSION_CHANGE,
    Chunk,
    ConflictPair,
    Document,
)
from services.version_resolver import version_sort_key

log = logging.getLogger(__name__)

router = APIRouter(tags=["conflicts"])

Status = Literal["needs_review", "resolved", "version_change"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChunkInfo(BaseModel):
    chunk_id: str
    doc_id: str
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
    needs_review: bool
    is_resolved: bool
    resolution_type: Optional[str] = None   # "prefer_source" | "both_valid"
    preferred_chunk_id: Optional[str] = None
    resolution_note: Optional[str] = None
    resolved_at: Optional[str] = None
    # For a version change, chunk_a is the earlier version and chunk_b the
    # later one. For a contradiction the order carries no meaning.
    chunk_a: ChunkInfo
    chunk_b: ChunkInfo


class ConflictsResponse(BaseModel):
    total: int          # matching the current filters
    page: int
    limit: int
    conflicts: list[ConflictRecord]
    # Corpus-wide counts, independent of filters and paging.
    total_needs_review: int = 0
    total_resolved: int = 0
    total_version_changes: int = 0


class ResolveRequest(BaseModel):
    resolution_type: Literal["prefer_source", "both_valid"]
    preferred_chunk_id: Optional[str] = None
    resolution_note: str = Field(default="", max_length=2000)


# ---------------------------------------------------------------------------
# GET /api/v1/conflicts
# ---------------------------------------------------------------------------


def _status_filter(status: Status):
    if status == "version_change":
        return ConflictPair.conflict_type == VERSION_CHANGE
    resolved = status == "resolved"
    return (ConflictPair.conflict_type == CONTRADICTION) & (ConflictPair.is_resolved.is_(resolved))


@router.get("/conflicts", response_model=ConflictsResponse)
def list_conflicts(
    status: Optional[Status] = Query(default=None, description="needs_review | resolved | version_change"),
    detected_during: Optional[str] = Query(default=None, description="Filter by query_id"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> ConflictsResponse:
    """List detected conflict pairs with both chunk snippets, newest first."""
    q = db.query(ConflictPair)
    if status is not None:
        q = q.filter(_status_filter(status))
    if detected_during:
        q = q.filter(ConflictPair.detected_during == detected_during)

    total = q.count()
    rows: list[ConflictPair] = (
        q.order_by(ConflictPair.detected_at.desc(), ConflictPair.conflict_id)
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    records = [r for r in (_record(db, row) for row in rows) if r is not None]

    def count(status: Status) -> int:
        return db.query(func.count(ConflictPair.conflict_id)).filter(_status_filter(status)).scalar() or 0

    return ConflictsResponse(
        total=total,
        page=page,
        limit=limit,
        conflicts=records,
        total_needs_review=count("needs_review"),
        total_resolved=count("resolved"),
        total_version_changes=count("version_change"),
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
    Record a reviewer's decision on a contradiction, replacing any earlier one.

    belief_revision.py applies it on every later query that retrieves both
    chunks: for questions about the present, prefer_source excludes the other
    side; for questions about the past both sides are kept and the decision is
    explained to the model.
    """
    row: ConflictPair | None = (
        db.query(ConflictPair).filter(ConflictPair.conflict_id == conflict_id).first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Conflict {conflict_id!r} not found.")

    if row.conflict_type != CONTRADICTION:
        raise HTTPException(
            status_code=409,
            detail="This is a change between versions of one document. Its version "
                   "history already explains it, so it is not resolved by review.",
        )

    pair = (row.chunk_id_a, row.chunk_id_b)
    if body.resolution_type == PREFER_SOURCE and body.preferred_chunk_id not in pair:
        raise HTTPException(
            status_code=422,
            detail="prefer_source needs preferred_chunk_id set to one of the two chunks in this conflict.",
        )
    if body.resolution_type == BOTH_VALID and body.preferred_chunk_id is not None:
        raise HTTPException(
            status_code=422,
            detail="both_valid prefers neither side, so preferred_chunk_id must be omitted.",
        )

    row.is_resolved = True
    row.resolution_type = body.resolution_type
    row.preferred_chunk_id = body.preferred_chunk_id
    row.resolution_note = body.resolution_note.strip() or None
    row.resolved_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(row)
    except Exception as exc:
        db.rollback()
        log.error("Failed to resolve conflict %s: %s", conflict_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while resolving conflict.")

    record = _record(db, row)
    if record is None:
        raise HTTPException(status_code=500, detail="Chunk data unavailable for this conflict.")
    return record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(db: Session, row: ConflictPair) -> ConflictRecord | None:
    """The API view of a row, or None if either chunk no longer exists."""
    a = _fetch_chunk_info(db, row.chunk_id_a)
    b = _fetch_chunk_info(db, row.chunk_id_b)
    if a is None or b is None:
        return None
    if row.conflict_type == VERSION_CHANGE and version_sort_key(b.version_string) < version_sort_key(a.version_string):
        # Versions of one lineage strictly increase, so this orders the pair
        # earlier → later. It is only meaningful within a lineage.
        a, b = b, a
    return ConflictRecord(
        conflict_id=row.conflict_id,
        conflict_type=row.conflict_type or "unknown",
        nli_score=round(row.nli_score or 0.0, 4),
        detected_at=_iso(row.detected_at),
        detected_during=row.detected_during,
        needs_review=row.needs_review(),
        is_resolved=row.is_resolved,
        resolution_type=row.resolution_type,
        preferred_chunk_id=row.preferred_chunk_id,
        resolution_note=row.resolution_note,
        resolved_at=_iso(row.resolved_at) if row.resolved_at else None,
        chunk_a=a,
        chunk_b=b,
    )


def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


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
        doc_id=doc.doc_id,
        snippet=chunk.content_snippet or chunk.content[:250],
        section_heading=chunk.section_heading,
        doc_title=doc.title,
        version_string=doc.version_string,
        published_at=(doc.published_at.strftime("%Y-%m-%d") if doc.published_at else None),
        is_latest=bool(doc.is_latest),
    )
