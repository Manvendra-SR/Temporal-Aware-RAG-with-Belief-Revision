"""
models.py — Full SQLAlchemy ORM models for all phases.

Tables are created with Base.metadata.create_all() at startup.
Columns that aren't populated yet simply hold NULL — no migrations needed.

Phase 1: Document, Chunk
Phase 3: QueryLog         (added here now so create_all handles it automatically)
Phase 6: ConflictPair     (added here now)
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(Base):
    """Represents a single ingested source document (PDF, MD, TXT).

    A document can belong to a lineage: parent_doc_id points to the
    immediately preceding version. is_latest=True marks the most recent.
    """

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # "pdf" | "md" | "txt"

    # Temporal / version metadata (Phase 4+)
    version_string: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    # Lineage (Phase 4+)
    parent_doc_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.doc_id", ondelete="SET NULL"),
        nullable=True,
    )
    is_latest: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Deduplication
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)  # SHA-256

    # Arbitrary extra metadata (stored as JSON)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    parent: Mapped["Document | None"] = relationship(
        "Document", remote_side="Document.doc_id", foreign_keys=[parent_doc_id]
    )

    def __repr__(self) -> str:
        return f"<Document {self.doc_id[:8]} title={self.title!r} v={self.version_string}>"


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class Chunk(Base):
    """A text chunk derived from a Document.

    Each chunk is embedded and indexed in FAISS. The faiss_index_id column
    maps the chunk to its position in the FAISS index.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    doc_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # FAISS index reference (Phase 2+)
    faiss_index_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Position within the document
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_snippet: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )  # first ~200 chars
    section_heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Character offsets in the original parsed text (for diffing / highlighting)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Temporal validity (Phase 4+)
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    superseded_by: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("chunks.chunk_id", ondelete="SET NULL"),
        nullable=True,
    )

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    # Arbitrary extra metadata
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk {self.chunk_id[:8]} doc={self.doc_id[:8]} idx={self.chunk_index}>"


# ---------------------------------------------------------------------------
# QueryLog  (Phase 3 — table created at startup, populated in Phase 3)
# ---------------------------------------------------------------------------


class QueryLog(Base):
    """Records every query made to the system for evaluation and debugging."""

    __tablename__ = "query_log"

    query_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    queried_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    # Stored as JSON array of chunk_id strings
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conflicts_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<QueryLog {self.query_id[:8]} latency={self.latency_ms}ms>"


# ---------------------------------------------------------------------------
# ConflictPair  (Phase 6)
# ---------------------------------------------------------------------------


class ConflictPair(Base):
    """Records a detected contradiction between two chunks."""

    __tablename__ = "conflict_pairs"
    __table_args__ = (UniqueConstraint("chunk_id_a", "chunk_id_b"),)

    conflict_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=_uuid
    )
    chunk_id_a: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id_b: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
    )

    # "direct_contradiction" | "version_supersession" | "scope_change"
    conflict_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    nli_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    detected_during: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), nullable=True
    )  # query_id

    # Resolution (Phase 7+)
    is_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    resolution_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<ConflictPair {self.conflict_id[:8]} "
            f"type={self.conflict_type} score={self.nli_score:.2f}>"
        )
