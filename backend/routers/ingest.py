"""
routers/ingest.py — Document ingestion and listing routes.

Routes:
    POST /api/v1/ingest
    GET  /api/v1/documents
    GET  /api/v1/documents/roots
    GET  /api/v1/documents/{doc_id}
    GET  /api/v1/documents/{doc_id}/lineage
    DELETE /api/v1/documents/{doc_id}
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models import Chunk, Document
from services import bm25_store, embedder, faiss_store
from services.chunker import chunk as do_chunk
from services.parser import parse, ParseResult
from services.version_resolver import (
    apply_lineage,
    unlink_lineage,
    validate_lineage,
    version_sort_key,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^v?\d[\d._\-]*$")


def _validate_version(v: str) -> str:
    """Raise HTTPException 422 if version_string doesn't match the allowed format."""
    v = v.strip()
    if not v or not _VERSION_RE.match(v):
        raise HTTPException(
            status_code=422,
            detail=(
                "version_string must start with a digit or 'v' followed by digits "
                "(e.g. '2.2', 'v1.13.1', '2024-03-01'). Got: " + repr(v)
            ),
        )
    return v


def _validate_date(raw: str) -> datetime:
    """Parse a strict YYYY-MM-DD date string and return a UTC-aware datetime."""
    raw = raw.strip()
    try:
        d = date.fromisoformat(raw)   # raises ValueError for anything not YYYY-MM-DD
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"published_at must be a valid date in YYYY-MM-DD format. Got: {raw!r}",
        )
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    version_string: str
    published_at: str
    is_latest: bool
    chunks_created: int
    ingested_at: str
    lineage_message: str


class DeleteResponse(BaseModel):
    doc_id: str
    title: str
    version_string: Optional[str]
    chunks_deleted: int
    lineage_message: str


class ChunkSummary(BaseModel):
    chunk_id: str
    chunk_index: int
    section_heading: Optional[str]
    content_snippet: str
    token_count: Optional[int]


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_type: Optional[str]
    chunk_count: int
    ingested_at: str
    checksum: Optional[str]
    version_string: Optional[str]
    published_at: Optional[str]
    is_latest: bool
    parent_doc_id: Optional[str]


class DocumentDetail(BaseModel):
    doc_id: str
    title: str
    source_type: Optional[str]
    ingested_at: str
    version_string: Optional[str]
    published_at: Optional[str]
    is_latest: bool
    parent_doc_id: Optional[str]
    chunks: list[ChunkSummary]


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int
    limit: int


class DocumentRoot(BaseModel):
    """Minimal representation of a document for the 'New Version' parent selector."""
    doc_id: str
    title: str
    version_string: Optional[str]


class LineageEntry(BaseModel):
    doc_id: str
    title: str
    version_string: Optional[str]
    published_at: Optional[str]
    ingested_at: str
    is_latest: bool


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    version_string: str = Form(...),
    published_at: str = Form(...),             # YYYY-MM-DD
    parent_doc_id: Optional[str] = Form(None), # explicit lineage; None = new document
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Full ingestion pipeline:
        1. Validate version_string and published_at formats
        2. Validate the lineage (parent exists, is latest, version and date
           increase) — read-only, before any expensive work
        3. Read file bytes and SHA-256 duplicate check
        4. Parse → text + headings
        5. Chunk → ChunkData list
        6. Embed all chunks
        7. Write Document + Chunks to PostgreSQL
        8. Apply the lineage mutation (supersede the parent), re-checking the
           same guards under the parent's row lock
        9. COMMIT, then add vectors to the FAISS + BM25 indexes

    Ordering note (lineage): the guards in step 2 need only the submitted
    metadata and the parent row, never the new document, so they can run
    before parsing, chunking and embedding instead of after — a rejected
    version no longer costs that work. Step 8 remains authoritative: the early
    check can be invalidated by a concurrent ingest of another version of the
    same parent, so the guards run again inside the transaction while holding
    the parent's row lock.

    Ordering note (indexes): the indexes are written LAST, after the database commit.
    They used to be written first, which meant any failure in steps 7-8 — such
    as the lineage re-check rejecting a version a concurrent ingest just
    overtook — rolled back the database but left the chunks permanently in
    FAISS and BM25.
    Those orphans are then retrieved but cannot be resolved back to a row, so
    they silently shrink every future result set. Postgres is the source of
    truth and the indexes are derivable from it (scripts/rebuild_indexes.py),
    so committing first makes the failure mode recoverable in the right
    direction.
    """
    # ── 1. Validate metadata ─────────────────────────────────────────────────
    version_string = _validate_version(version_string)
    published_at_dt = _validate_date(published_at)

    # ── 2. Validate lineage (read-only, before any expensive work) ───────────
    try:
        validate_lineage(
            parent_doc_id=parent_doc_id,
            new_version=version_string,
            new_date=published_at_dt,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── 3. Read file + duplicate check ───────────────────────────────────────
    file_bytes = await file.read()
    filename = file.filename or "upload"

    checksum = hashlib.sha256(file_bytes).hexdigest()
    existing = db.query(Document).filter(Document.checksum == checksum).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate file: already ingested as doc_id={existing.doc_id!r}",
        )

    # ── 4. Parse ────────────────────────────────────────────────────────────
    try:
        parsed: ParseResult = parse(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    log.info("Parsed '%s': %d chars, %d headings.", filename, len(parsed.text), len(parsed.headings))

    # ── 5. Chunk ────────────────────────────────────────────────────────────
    chunks = do_chunk(parsed.text, parsed.headings, embedder=embedder)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text content could be extracted from the file.")

    log.info("Chunked into %d chunks.", len(chunks))

    # ── 6. Embed ────────────────────────────────────────────────────────────
    vectors = embedder.embed([c.content for c in chunks])  # (N, 384)

    # Reserve FAISS ids and chunk UUIDs. next_id() only advances an in-memory
    # counter; nothing is written to either index until after the commit below.
    faiss_ids = [faiss_store.next_id() for _ in chunks]
    chunk_uuids = [str(uuid.uuid4()) for _ in chunks]

    # ── 7. Write to PostgreSQL ──────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    doc_id = str(uuid.uuid4())

    doc = Document(
        doc_id=doc_id,
        title=title,
        source_type=parsed.source_type,
        checksum=checksum,
        ingested_at=now,
        version_string=version_string,
        published_at=published_at_dt,
        is_latest=True,  # may be overridden by the lineage mutation
    )
    db.add(doc)
    db.flush()  # get doc_id into DB before adding chunks

    db_chunks = []
    for chunk_data, faiss_id, chunk_uuid in zip(chunks, faiss_ids, chunk_uuids):
        db_chunks.append(Chunk(
            chunk_id=chunk_uuid,
            doc_id=doc_id,
            faiss_index_id=faiss_id,
            chunk_index=chunk_data.chunk_index,
            content=chunk_data.content,
            content_snippet=chunk_data.content_snippet,
            section_heading=chunk_data.section_heading,
            token_count=chunk_data.token_count,
            char_start=chunk_data.char_start,
            char_end=chunk_data.char_end,
            ingested_at=now,
            valid_from=published_at_dt,
        ))

    db.bulk_save_objects(db_chunks)
    # Don't commit yet — the lineage mutation belongs in the same transaction

    # ── 8. Apply the lineage mutation (authoritative re-check) ──────────────
    try:
        lineage = apply_lineage(
            parent_doc_id=parent_doc_id,
            new_doc_id=doc_id,
            new_version=version_string,
            new_date=published_at_dt,
            db=db,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))

    # Apply lineage result to the new document
    doc.parent_doc_id = lineage.parent_doc_id

    db.commit()

    # ── 9. Index (only now that the rows are durably committed) ─────────────
    # If this fails the document exists but is not searchable — a recoverable
    # state, fixable with scripts/rebuild_indexes.py. The reverse ordering
    # would leave unreachable vectors that no rebuild can clean up.
    # BM25 indexes `content` — the same text that was embedded, and the same
    # text persisted to the chunks table. It previously indexed `raw_content`,
    # which is never stored, so the BM25 index could not be rebuilt from the
    # database and the two retrievers searched subtly different text.
    try:
        faiss_store.add(faiss_ids, vectors)
        bm25_store.add(chunk_uuids, [c.content for c in chunks])
    except Exception:
        log.exception(
            "Indexing failed for doc_id=%s after the database commit. The "
            "document is stored but will not be retrievable until the indexes "
            "are rebuilt (python scripts/rebuild_indexes.py).",
            doc_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Document was saved but could not be indexed, so it is not yet "
                "searchable. Rebuild the search indexes to recover it."
            ),
        )

    log.info(
        "Ingested doc_id=%s with %d chunks. Lineage: %s",
        doc_id, len(chunks), lineage.lineage_message,
    )

    return IngestResponse(
        doc_id=doc_id,
        title=title,
        version_string=version_string,
        published_at=published_at_dt.date().isoformat(),
        is_latest=doc.is_latest,
        chunks_created=len(chunks),
        ingested_at=now.isoformat(),
        lineage_message=lineage.lineage_message,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents/roots
# ---------------------------------------------------------------------------


@router.get("/documents/roots", response_model=list[DocumentRoot])
def list_latest_documents(db: Session = Depends(get_db)) -> list[DocumentRoot]:
    """
    Return all is_latest=True documents for the 'New Version' parent selector.
    The frontend uses this to populate the searchable dropdown when a user
    chooses 'Upload New Version of Existing Document'.
    """
    docs = (
        db.query(Document)
        .filter(Document.is_latest == True)  # noqa: E712
        .order_by(Document.title)
        .all()
    )
    return [
        DocumentRoot(
            doc_id=d.doc_id,
            title=d.title,
            version_string=d.version_string,
        )
        for d in docs
    ]


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Return a paginated list of ingested documents."""
    query = db.query(Document)

    total = query.count()
    docs = query.order_by(Document.ingested_at.desc()).offset((page - 1) * limit).limit(limit).all()

    items = []
    for doc in docs:
        chunk_count = db.query(Chunk).filter(Chunk.doc_id == doc.doc_id).count()
        items.append(DocumentSummary(
            doc_id=doc.doc_id,
            title=doc.title,
            source_type=doc.source_type,
            chunk_count=chunk_count,
            ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else "",
            checksum=doc.checksum,
            version_string=doc.version_string,
            published_at=doc.published_at.isoformat() if doc.published_at else None,
            is_latest=doc.is_latest,
            parent_doc_id=doc.parent_doc_id,
        ))

    return DocumentListResponse(items=items, total=total, page=page, limit=limit)


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{doc_id}
# ---------------------------------------------------------------------------


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str, db: Session = Depends(get_db)) -> DocumentDetail:
    """Return a document with its full chunk list."""
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    db_chunks = (
        db.query(Chunk)
        .filter(Chunk.doc_id == doc_id)
        .order_by(Chunk.chunk_index)
        .all()
    )

    return DocumentDetail(
        doc_id=doc.doc_id,
        title=doc.title,
        source_type=doc.source_type,
        ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else "",
        version_string=doc.version_string,
        published_at=doc.published_at.isoformat() if doc.published_at else None,
        is_latest=doc.is_latest,
        parent_doc_id=doc.parent_doc_id,
        chunks=[
            ChunkSummary(
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                section_heading=c.section_heading,
                content_snippet=c.content_snippet or c.content[:200],
                token_count=c.token_count,
            )
            for c in db_chunks
        ],
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{doc_id}/lineage
# ---------------------------------------------------------------------------


@router.get("/documents/{doc_id}/lineage", response_model=list[LineageEntry])
def get_lineage(doc_id: str, db: Session = Depends(get_db)) -> list[LineageEntry]:
    """
    Return the full lineage chain for a document, ordered oldest → newest.
    Traverses parent_doc_id links to find the root, then returns all siblings
    belonging to the same lineage chain.
    """
    # Find the target document
    target = db.query(Document).filter(Document.doc_id == doc_id).first()
    if target is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Walk UP to find the root (doc with no parent)
    root = target
    visited: set[str] = set()
    while root.parent_doc_id and root.parent_doc_id not in visited:
        visited.add(root.doc_id)
        parent = db.query(Document).filter(Document.doc_id == root.parent_doc_id).first()
        if parent is None:
            break
        root = parent

    # Now walk DOWN from the root collecting all descendants
    chain: list[Document] = [root]
    frontier = [root]
    seen: set[str] = {root.doc_id}

    while frontier:
        next_frontier: list[Document] = []
        for node in frontier:
            children = (
                db.query(Document)
                .filter(Document.parent_doc_id == node.doc_id)
                .all()
            )
            for child in children:
                if child.doc_id not in seen:
                    seen.add(child.doc_id)
                    chain.append(child)
                    next_frontier.append(child)
        frontier = next_frontier

    # Sort by version (oldest first), falling back to ingested_at for ties.
    # Uses the shared ordering policy from version_resolver so the lineage strip
    # in the UI orders versions exactly the way the lineage guard does.
    chain.sort(key=lambda d: (
        version_sort_key(d.version_string),
        d.ingested_at or datetime.min.replace(tzinfo=timezone.utc),
    ))

    return [
        LineageEntry(
            doc_id=d.doc_id,
            title=d.title,
            version_string=d.version_string,
            published_at=d.published_at.isoformat() if d.published_at else None,
            ingested_at=d.ingested_at.isoformat() if d.ingested_at else "",
            is_latest=d.is_latest,
        )
        for d in chain
    ]


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{doc_id}
# ---------------------------------------------------------------------------


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    """
    Delete a document, its chunks, and its place in the version lineage.

    Lineage rule: a document can only be deleted if no other version names it
    as its parent. For a chain v1 → v2 → v3 that means v3 can be deleted and
    v1 and v2 cannot, because the version after them depends on them. Deleting
    v3 restores v2 as the latest version and reopens its chunks' validity
    windows — see version_resolver.unlink_lineage(). A document with
    descendants is refused with a 409 and nothing is written.

    The whole thing is one transaction, and the target row is held FOR UPDATE
    from the first read, so a concurrent ingest cannot add a child between the
    dependency check and the delete: whichever transaction commits first wins,
    and the other is rejected (409 here, 422 there).

    Chunks: removed by the database, via the ON DELETE CASCADE on
    chunks.doc_id; conflict_pairs cascade from the chunks in turn.

    Indexes: the chunks' vectors are removed from FAISS by their
    faiss_index_id and their text from the BM25 corpus, both exactly — the
    FAISS index is a flat inner-product index, which supports remove_ids().
    Both are updated AFTER the database commit, deliberately. The two stores
    cannot be committed atomically with Postgres, so the ordering decides
    which way an interrupted delete fails: this way the rows are gone and at
    worst some index entries outlive them, which retrieval ignores (a hit that
    resolves to no row is skipped) and rebuild_indexes.py cleans up. Removing
    from the indexes first would mean a failed commit leaves a document that
    still exists but can no longer be found — silent and much harder to
    notice.
    """
    doc = (
        db.query(Document)
        .filter(Document.doc_id == doc_id)
        .with_for_update()
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        lineage_message = unlink_lineage(doc, db)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))

    # Read the index keys while the chunks still exist; the cascade takes the
    # rows with the document and there is no second chance to look them up.
    rows = (
        db.query(Chunk.chunk_id, Chunk.faiss_index_id)
        .filter(Chunk.doc_id == doc_id)
        .all()
    )
    chunk_ids = [row.chunk_id for row in rows]
    faiss_ids = [row.faiss_index_id for row in rows if row.faiss_index_id is not None]
    title, version_string = doc.title, doc.version_string

    db.delete(doc)
    db.commit()

    # Index cleanup last, after the rows are durably gone — the same ordering
    # as ingestion, and for the same reason: Postgres is the source of truth
    # and the indexes are derivable from it.
    try:
        faiss_store.remove(faiss_ids)
        bm25_store.remove(chunk_ids)
    except Exception:
        log.exception(
            "Failed to remove the %d chunks of deleted doc_id=%s from the search "
            "indexes. They cannot be retrieved (retrieval resolves every hit "
            "through the database), but the indexes should be rebuilt to drop "
            "them (python scripts/rebuild_indexes.py).",
            len(chunk_ids), doc_id,
        )

    log.info("Deleted doc_id=%s with %d chunks. Lineage: %s", doc_id, len(chunk_ids), lineage_message)

    return DeleteResponse(
        doc_id=doc_id,
        title=title,
        version_string=version_string,
        chunks_deleted=len(chunk_ids),
        lineage_message=lineage_message,
    )
