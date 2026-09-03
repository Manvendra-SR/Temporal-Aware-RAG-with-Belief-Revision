"""
routers/analytics.py — Read-only aggregate endpoints.

Routes:
    GET /api/v1/analytics/overview   — corpus counts for the sidebar
    GET /api/v1/analytics/timeline   — version lineage graph for the Library page

On the removed "domain" concept
-------------------------------
Earlier revisions grouped these results by `Document.metadata_json["domain"]`.
No ingestion path has ever written that key — there is no `domain` column, no
form field, and no UI for one — so every grouping collapsed to a single bucket
labelled "unknown" and the freshness endpoint reported one meaningless row.
Rather than surface a permanently empty dimension, documents are now grouped by
their version LINEAGE, which is real data the ingest flow actually produces.

Endpoints for per-domain freshness, conflict-type distribution and recent
queries were removed along with the unrouted AnalyticsPage that was their only
consumer. Query history is available directly from the `query_log` table and
should come back through the evaluation harness, which needs it anyway.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models import Chunk, ConflictPair, Document, QueryLog
from services.version_resolver import version_sort_key

log = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class OverviewResponse(BaseModel):
    total_docs: int
    total_chunks: int
    total_conflicts: int
    unresolved_conflicts: int
    total_queries: int
    # Number of distinct version lineages (a doc and all its versions = one)
    total_lineages: int
    # Documents superseded by a newer version
    superseded_docs: int


class TimelineNode(BaseModel):
    doc_id: str
    title: str
    version_string: Optional[str]
    published_at: Optional[str]
    is_latest: bool
    parent_doc_id: Optional[str]
    chunk_count: int
    unresolved_conflicts: int
    # Identifies which lineage this document belongs to — the doc_id of the
    # oldest ancestor. Documents sharing a lineage_id form one version chain.
    lineage_id: str
    lineage_title: str
    # "latest" | "superseded" | "conflict"
    status: str


class TimelineResponse(BaseModel):
    nodes: list[TimelineNode]


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/overview
# ---------------------------------------------------------------------------


@router.get("/overview", response_model=OverviewResponse)
def analytics_overview(db: Session = Depends(get_db)) -> OverviewResponse:
    """High-level corpus counts, shown in the sidebar."""
    total_docs = db.query(func.count(Document.doc_id)).scalar() or 0
    total_chunks = db.query(func.count(Chunk.chunk_id)).scalar() or 0
    total_conflicts = db.query(func.count(ConflictPair.conflict_id)).scalar() or 0
    unresolved = (
        db.query(func.count(ConflictPair.conflict_id))
        .filter(ConflictPair.is_resolved.is_(False))
        .scalar() or 0
    )
    total_queries = db.query(func.count(QueryLog.query_id)).scalar() or 0

    # A lineage is a chain of versions; its root is a document with no parent.
    total_lineages = (
        db.query(func.count(Document.doc_id))
        .filter(Document.parent_doc_id.is_(None))
        .scalar() or 0
    )
    superseded_docs = (
        db.query(func.count(Document.doc_id))
        .filter(Document.is_latest.is_(False))
        .scalar() or 0
    )

    return OverviewResponse(
        total_docs=total_docs,
        total_chunks=total_chunks,
        total_conflicts=total_conflicts,
        unresolved_conflicts=unresolved,
        total_queries=total_queries,
        total_lineages=total_lineages,
        superseded_docs=superseded_docs,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/timeline
# ---------------------------------------------------------------------------


@router.get("/timeline", response_model=TimelineResponse)
def analytics_timeline(db: Session = Depends(get_db)) -> TimelineResponse:
    """
    Return every document as a timeline node, grouped into version lineages.

    status:
      "conflict"   — the document has at least one unresolved conflict
      "latest"     — is_latest=True and no unresolved conflicts
      "superseded" — is_latest=False
    """
    docs = db.query(Document).all()
    if not docs:
        return TimelineResponse(nodes=[])

    by_id = {d.doc_id: d for d in docs}

    # Chunk count per doc
    chunk_counts: dict[str, int] = {
        row.doc_id: row.cnt
        for row in db.query(
            Chunk.doc_id,
            func.count(Chunk.chunk_id).label("cnt"),
        ).group_by(Chunk.doc_id).all()
    }

    # Unresolved conflicts per doc. Only the chunks referenced by an unresolved
    # conflict are loaded, rather than the whole chunks table as before.
    unresolved_pairs = (
        db.query(ConflictPair.chunk_id_a, ConflictPair.chunk_id_b)
        .filter(ConflictPair.is_resolved.is_(False))
        .all()
    )
    conflicted_chunk_ids = {
        cid for pair in unresolved_pairs for cid in (pair.chunk_id_a, pair.chunk_id_b)
    }
    unresolved_per_doc: dict[str, int] = {}
    if conflicted_chunk_ids:
        rows = (
            db.query(Chunk.chunk_id, Chunk.doc_id)
            .filter(Chunk.chunk_id.in_(conflicted_chunk_ids))
            .all()
        )
        chunk_to_doc = {r.chunk_id: r.doc_id for r in rows}
        for pair in unresolved_pairs:
            for cid in (pair.chunk_id_a, pair.chunk_id_b):
                doc_id = chunk_to_doc.get(cid)
                if doc_id:
                    unresolved_per_doc[doc_id] = unresolved_per_doc.get(doc_id, 0) + 1

    nodes: list[TimelineNode] = []
    for doc in docs:
        lineage_root = _find_lineage_root(doc, by_id)
        unresolved = unresolved_per_doc.get(doc.doc_id, 0)

        if unresolved > 0:
            status = "conflict"
        elif doc.is_latest:
            status = "latest"
        else:
            status = "superseded"

        nodes.append(TimelineNode(
            doc_id=doc.doc_id,
            title=doc.title,
            version_string=doc.version_string,
            published_at=(
                doc.published_at.strftime("%Y-%m-%d") if doc.published_at else None
            ),
            is_latest=doc.is_latest,
            parent_doc_id=doc.parent_doc_id,
            chunk_count=chunk_counts.get(doc.doc_id, 0),
            unresolved_conflicts=unresolved,
            lineage_id=lineage_root.doc_id,
            lineage_title=lineage_root.title,
            status=status,
        ))

    # Oldest version first within each lineage, using the shared ordering policy.
    nodes.sort(key=lambda n: (n.lineage_title, version_sort_key(n.version_string)))
    return TimelineResponse(nodes=nodes)


def _find_lineage_root(doc: Document, by_id: dict[str, Document]) -> Document:
    """
    Walk parent_doc_id links to the oldest ancestor.

    Guards against cycles (a corrupted parent chain would otherwise hang the
    request) by tracking visited ids and stopping if one repeats.
    """
    current = doc
    seen: set[str] = {current.doc_id}
    while current.parent_doc_id:
        parent = by_id.get(current.parent_doc_id)
        if parent is None or parent.doc_id in seen:
            break
        seen.add(parent.doc_id)
        current = parent
    return current
