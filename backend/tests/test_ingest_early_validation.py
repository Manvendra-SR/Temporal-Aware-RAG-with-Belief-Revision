"""
The ingestion route rejects a bad lineage before it does any expensive work.

Parsing, chunking and embedding a document is the costly part of ingestion,
and a version/date/parent guard needs none of it — only the submitted metadata
and the parent row. These tests drive POST /api/v1/ingest against an in-memory
SQLite database with parse/chunk/embed replaced by counting spies, so they can
assert the stronger property: not just that the request fails with 422, but
that the pipeline never got as far as touching the file.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker

import models  # noqa: F401 — registers the tables on Base
from database import Base, get_db
from models import Chunk, Document
from routers import ingest as ingest_router
from services.chunker import ChunkData


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


class Spies:
    """Counts how far into the pipeline a request got."""

    def __init__(self) -> None:
        self.parsed = 0
        self.chunked = 0
        self.embedded = 0
        self.indexed = 0

    @property
    def did_expensive_work(self) -> bool:
        return bool(self.parsed or self.chunked or self.embedded)


@pytest.fixture
def db():
    # StaticPool + check_same_thread: TestClient runs the route in another
    # thread, which would otherwise get its own empty in-memory database.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def spies(monkeypatch) -> Spies:
    s = Spies()

    def fake_parse(file_bytes, filename):
        s.parsed += 1
        return ingest_router.ParseResult(text="some text", headings=[], source_type="txt")

    def fake_chunk(text, headings, embedder=None):
        s.chunked += 1
        return [ChunkData(
            chunk_index=0, content=text, raw_content=text, content_snippet=text[:200],
            section_heading=None, token_count=2, char_start=0, char_end=len(text),
        )]

    def fake_embed(texts):
        s.embedded += 1
        return np.zeros((len(texts), 384), dtype="float32")

    def fake_faiss_add(ids, vectors, persist=True):
        s.indexed += 1

    def fake_bm25_add(ids, texts):
        s.indexed += 1

    monkeypatch.setattr(ingest_router, "parse", fake_parse)
    monkeypatch.setattr(ingest_router, "do_chunk", fake_chunk)
    monkeypatch.setattr(ingest_router.embedder, "embed", fake_embed)
    monkeypatch.setattr(ingest_router.faiss_store, "add", fake_faiss_add)
    monkeypatch.setattr(ingest_router.faiss_store, "next_id", lambda: 0)
    monkeypatch.setattr(ingest_router.bm25_store, "add", fake_bm25_add)
    return s


@pytest.fixture
def client(db, spies):
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c


def upload(client, *, version: str, published: str, parent: str | None = None, body: str = "x"):
    data = {"title": "Handbook", "version_string": version, "published_at": published}
    if parent is not None:
        data["parent_doc_id"] = parent
    return client.post(
        "/api/v1/ingest",
        data=data,
        files={"file": (f"{version}-{body}.txt", body.encode(), "text/plain")},
    )


@pytest.fixture
def parent_doc(db) -> Document:
    doc = Document(title="Handbook", version_string="2.0", published_at=utc(2023, 1, 1),
                   is_latest=True)
    db.add(doc)
    db.flush()  # assigns doc_id
    db.add(Chunk(doc_id=doc.doc_id, chunk_index=0, content="v2 text",
                 valid_from=utc(2023, 1, 1)))
    db.commit()
    return doc


class TestEarlyRejection:
    def test_unknown_parent(self, client, spies) -> None:
        r = upload(client, version="2.0", published="2024-01-01",
                   parent="00000000-0000-0000-0000-000000000000")
        assert r.status_code == 422
        assert "not found" in r.json()["detail"]
        assert not spies.did_expensive_work

    def test_non_latest_parent(self, client, db, spies, parent_doc) -> None:
        parent_doc.is_latest = False
        db.commit()
        r = upload(client, version="3.0", published="2024-01-01", parent=parent_doc.doc_id)
        assert r.status_code == 422
        assert "not the latest version" in r.json()["detail"]
        assert not spies.did_expensive_work

    def test_non_increasing_version(self, client, spies, parent_doc) -> None:
        r = upload(client, version="1.5", published="2024-01-01", parent=parent_doc.doc_id)
        assert r.status_code == 422
        assert "strictly greater" in r.json()["detail"]
        assert not spies.did_expensive_work

    def test_earlier_publication_date(self, client, spies, parent_doc) -> None:
        r = upload(client, version="3.0", published="2022-06-01", parent=parent_doc.doc_id)
        assert r.status_code == 422
        assert "earlier than the parent" in r.json()["detail"]
        assert not spies.did_expensive_work

    def test_nothing_is_written(self, client, db, spies, parent_doc) -> None:
        upload(client, version="1.5", published="2024-01-01", parent=parent_doc.doc_id)
        assert db.query(Document).count() == 1
        assert db.query(Chunk).count() == 1
        db.refresh(parent_doc)
        assert parent_doc.is_latest is True


class TestValidSubmissions:
    def test_a_new_lineage_runs_the_whole_pipeline(self, client, db, spies) -> None:
        r = upload(client, version="1.0", published="2024-01-01")
        assert r.status_code == 201, r.text
        assert (spies.parsed, spies.chunked, spies.embedded) == (1, 1, 1)
        assert r.json()["lineage_message"] == "New document lineage started."
        assert db.query(Document).count() == 1

    def test_a_new_version_supersedes_its_parent(self, client, db, spies, parent_doc) -> None:
        r = upload(client, version="3.0", published="2024-01-01", parent=parent_doc.doc_id)
        assert r.status_code == 201, r.text
        assert (spies.parsed, spies.chunked, spies.embedded) == (1, 1, 1)

        db.refresh(parent_doc)
        assert parent_doc.is_latest is False
        old_chunk = db.query(Chunk).filter(Chunk.doc_id == parent_doc.doc_id).one()
        assert old_chunk.is_superseded is True
        assert old_chunk.valid_to.replace(tzinfo=None) == datetime(2024, 1, 1)

        new_doc = db.query(Document).filter(Document.doc_id == r.json()["doc_id"]).one()
        assert (new_doc.parent_doc_id, new_doc.is_latest) == (parent_doc.doc_id, True)
