"""
routers/ingest.py — Document ingestion and listing routes.

Routes:
    POST /api/v1/ingest
    GET  /api/v1/documents
    GET  /api/v1/documents/roots
    GET  /api/v1/documents/{doc_id}
    GET  /api/v1/documents/{doc_id}/lineage
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
from services.version_resolver import resolve as resolve_lineage

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
        2. Read file bytes and SHA-256 duplicate check
        3. Parse → text + headings
        4. Chunk → ChunkData list
        5. Embed all chunks
        6. Add to FAISS + BM25 indexes
        7. Write Document + Chunks to PostgreSQL
        8. Resolve version lineage (explicit via parent_doc_id)
    """
    # ── 1. Validate metadata ─────────────────────────────────────────────────
    version_string = _validate_version(version_string)
    published_at_dt = _validate_date(published_at)

    # ── 2. Read file + duplicate check ───────────────────────────────────────
    file_bytes = await file.read()
    filename = file.filename or "upload"

    checksum = hashlib.sha256(file_bytes).hexdigest()
    existing = db.query(Document).filter(Document.checksum == checksum).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate file: already ingested as doc_id={existing.doc_id!r}",
        )

    # ── 3. Parse ────────────────────────────────────────────────────────────
    try:
        parsed: ParseResult = parse(file_bytes, filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    log.info("Parsed '%s': %d chars, %d headings.", filename, len(parsed.text), len(parsed.headings))

    # ── 4. Chunk ────────────────────────────────────────────────────────────
    chunks = do_chunk(parsed.text, parsed.headings)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text content could be extracted from the file.")

    log.info("Chunked into %d chunks.", len(chunks))

    # ── 5. Embed ────────────────────────────────────────────────────────────
    vectors = embedder.embed([c.content for c in chunks])  # (N, 384)

    # ── 6. Assign FAISS IDs ─────────────────────────────────────────────────
    faiss_ids = [faiss_store.next_id() for _ in chunks]
    faiss_store.add(faiss_ids, vectors)

    chunk_uuids = [str(uuid.uuid4()) for _ in chunks]
    bm25_store.add(chunk_uuids, [c.raw_content for c in chunks])

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
        is_latest=True,  # may be overridden by lineage resolver
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
            ingested_at=now,
            valid_from=published_at_dt,
        ))

    db.bulk_save_objects(db_chunks)
    # Don't commit yet — version resolver needs the session open

    # ── 8. Resolve version lineage ──────────────────────────────────────────
    try:
        lineage = resolve_lineage(
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

    # Sort by version (oldest first), fall back to ingested_at
    from services.version_resolver import _parse_version
    chain.sort(key=lambda d: (
        _parse_version(d.version_string)[0],
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
