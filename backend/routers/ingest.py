"""
routers/ingest.py — Document ingestion and listing routes.

Routes:
    POST /api/v1/ingest
    GET  /api/v1/documents
    GET  /api/v1/documents/{doc_id}
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models import Chunk, Document
from services import bm25_store, embedder, faiss_store
from services.chunker import chunk as do_chunk
from services.parser import parse, ParseResult

log = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])

ALLOWED_DOMAINS = [
    "npm_docs", "python_docs", "pytorch_docs",
    "arxiv_cs", "legal", "general",
]

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    doc_id: str
    title: str
    domain: str
    chunks_created: int
    ingested_at: str


class ChunkSummary(BaseModel):
    chunk_id: str
    chunk_index: int
    section_heading: Optional[str]
    content_snippet: str
    token_count: Optional[int]


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    domain: str
    source_type: Optional[str]
    chunk_count: int
    ingested_at: str
    checksum: Optional[str]


class DocumentDetail(BaseModel):
    doc_id: str
    title: str
    domain: str
    source_type: Optional[str]
    ingested_at: str
    chunks: list[ChunkSummary]


class DocumentListResponse(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int
    limit: int


# ---------------------------------------------------------------------------
# POST /api/v1/ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    domain: str = Form("general"),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """
    Full ingestion pipeline:
        1. Read file bytes
        2. SHA-256 duplicate check
        3. Parse → text + headings
        4. Chunk → ChunkData list
        5. Embed all chunks
        6. Add to FAISS + BM25 indexes
        7. Write Document + Chunks to PostgreSQL
    """
    # ── 1. Read file ────────────────────────────────────────────────────────
    file_bytes = await file.read()
    filename = file.filename or "upload"

    # ── 2. Duplicate check ──────────────────────────────────────────────────
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

    # Add to BM25 (chunk_ids will be set after DB insert; we'll do a second pass)
    # We pre-generate chunk UUIDs so we can index them before the DB insert.
    chunk_uuids = [str(uuid.uuid4()) for _ in chunks]
    bm25_store.add(chunk_uuids, [c.raw_content for c in chunks])

    # ── 7. Write to PostgreSQL ──────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    doc_id = str(uuid.uuid4())

    doc = Document(
        doc_id=doc_id,
        title=title,
        domain=domain,
        source_type=parsed.source_type,
        checksum=checksum,
        ingested_at=now,
        is_latest=True,
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
        ))

    db.bulk_save_objects(db_chunks)
    db.commit()

    log.info("Ingested doc_id=%s with %d chunks.", doc_id, len(chunks))

    return IngestResponse(
        doc_id=doc_id,
        title=title,
        domain=domain,
        chunks_created=len(chunks),
        ingested_at=now.isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------

@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    domain: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Return a paginated list of ingested documents."""
    query = db.query(Document)
    if domain:
        query = query.filter(Document.domain == domain)

    total = query.count()
    docs = query.order_by(Document.ingested_at.desc()).offset((page - 1) * limit).limit(limit).all()

    items = []
    for doc in docs:
        chunk_count = db.query(Chunk).filter(Chunk.doc_id == doc.doc_id).count()
        items.append(DocumentSummary(
            doc_id=doc.doc_id,
            title=doc.title,
            domain=doc.domain,
            source_type=doc.source_type,
            chunk_count=chunk_count,
            ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else "",
            checksum=doc.checksum,
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
        domain=doc.domain,
        source_type=doc.source_type,
        ingested_at=doc.ingested_at.isoformat() if doc.ingested_at else "",
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
