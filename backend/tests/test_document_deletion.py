"""
Deleting a document has to leave the version chain valid, so these tests drive
DELETE /api/v1/documents/{doc_id} against a real (in-memory SQLite) database
built through the same ingest path the app uses, with real FAISS and BM25
indexes in a temporary directory.

The rule under test: a version can only be deleted if nothing points at it as
its parent. For v1 → v2 → v3 that means v3 is deletable and v1 and v2 are not,
and deleting v3 hands "latest" back to v2 and reopens its chunks' validity
windows — the exact inverse of what ingesting v3 did to them.

The second thing under test is that a deleted document leaves the search
indexes: its vectors are removed from FAISS by faiss_index_id and its text
from the BM25 corpus, and the documents around it stay searchable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import models  # noqa: F401 — registers the tables on Base
from database import Base, get_db
from models import Chunk, Document
from routers import ingest as ingest_router
from services import bm25_store, faiss_store
from services.embedder import EMBEDDING_DIM
from services.version_resolver import apply_lineage, validate_lineage


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


@pytest.fixture
def db():
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
def indexes(tmp_path, monkeypatch):
    """Real FAISS and BM25 singletons, persisting into tmp_path, not data/."""
    monkeypatch.setattr(faiss_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(faiss_store, "_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(faiss_store, "_META_PATH", tmp_path / "faiss_meta.json")
    monkeypatch.setattr(bm25_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(bm25_store, "_BM25_PATH", tmp_path / "bm25.pkl")
    faiss_store.reset()
    bm25_store.reset()
    yield
    faiss_store.reset()
    bm25_store.reset()


@pytest.fixture
def client(db, indexes):
    app = FastAPI()
    app.include_router(ingest_router.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c


def vector(seed: int) -> np.ndarray:
    """A distinct unit vector, standing in for an embedding of one chunk."""
    v = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    v[0, seed % EMBEDDING_DIM] = 1.0
    return v


def ingest(
    db,
    version: str,
    published: datetime,
    parent: Document | None = None,
    *,
    chunks: int = 1,
) -> Document:
    """
    The ingest pipeline's effects on the database and both indexes, without
    the parsing and embedding in between.
    """
    validate_lineage(parent.doc_id if parent else None, version, published, db)
    doc = Document(title="Handbook", version_string=version, published_at=published)
    db.add(doc)
    db.flush()

    for i in range(chunks):
        faiss_id = faiss_store.next_id()
        text = f"text of {version} part {i}"
        chunk = Chunk(doc_id=doc.doc_id, chunk_index=i, content=text,
                      valid_from=published, faiss_index_id=faiss_id)
        db.add(chunk)
        db.flush()
        faiss_store.add([faiss_id], vector(faiss_id), persist=False)
        bm25_store.add([chunk.chunk_id], [text], persist=False)

    lineage = apply_lineage(parent.doc_id if parent else None, doc.doc_id, version, published, db)
    doc.parent_doc_id = lineage.parent_doc_id
    db.commit()
    return doc


def faiss_ids_of(db, doc: Document) -> list[int]:
    return [
        row.faiss_index_id
        for row in db.query(Chunk.faiss_index_id).filter(Chunk.doc_id == doc.doc_id).all()
    ]


def searchable_faiss_ids(k: int = 50) -> set[int]:
    """Every id the index will actually return, across all the test vectors."""
    found: set[int] = set()
    for dim in range(12):
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[dim] = 1.0
        ids, sims = faiss_store.search(v, k=k)
        found.update(i for i, sim in zip(ids, sims) if sim > 0.5)
    return found


@pytest.fixture
def chain(db, indexes) -> tuple[Document, Document, Document]:
    """v1 → v2 → v3, with v3 the latest version."""
    v1 = ingest(db, "1.0", utc(2022, 1, 10))
    v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
    v3 = ingest(db, "3.0", utc(2025, 2, 1), parent=v2)
    return v1, v2, v3


def chunk_of(db, doc: Document) -> Chunk:
    return db.query(Chunk).filter(Chunk.doc_id == doc.doc_id).one()


class TestBasicDeletion:
    def test_a_standalone_document_is_deleted_with_its_chunks(self, client, db) -> None:
        doc = ingest(db, "1.0", utc(2024, 1, 1))
        r = client.delete(f"/api/v1/documents/{doc.doc_id}")

        assert r.status_code == 200, r.text
        assert r.json()["chunks_deleted"] == 1
        assert db.query(Document).count() == 0
        assert db.query(Chunk).count() == 0

    def test_deleting_a_nonexistent_document_is_a_404(self, client, db) -> None:
        r = client.delete("/api/v1/documents/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404
        assert r.json()["detail"] == "Document not found."


class TestLineagePolicy:
    def test_the_latest_version_can_be_deleted(self, client, db, chain) -> None:
        _, _, v3 = chain
        r = client.delete(f"/api/v1/documents/{v3.doc_id}")
        assert r.status_code == 200, r.text
        assert db.query(Document).filter(Document.doc_id == v3.doc_id).first() is None

    def test_a_version_with_a_descendant_is_rejected(self, client, db, chain) -> None:
        v1, v2, _ = chain
        for doomed in (v1, v2):
            r = client.delete(f"/api/v1/documents/{doomed.doc_id}")
            assert r.status_code == 409, r.text
            assert "cannot be deleted" in r.json()["detail"]

    def test_a_rejected_deletion_changes_nothing(self, client, db, chain) -> None:
        v1, v2, v3 = chain
        before = {
            d.doc_id: (d.is_latest, d.parent_doc_id, chunk_of(db, d).is_superseded)
            for d in (v1, v2, v3)
        }

        assert client.delete(f"/api/v1/documents/{v1.doc_id}").status_code == 409

        db.expire_all()
        after = {
            d.doc_id: (d.is_latest, d.parent_doc_id, chunk_of(db, d).is_superseded)
            for d in (v1, v2, v3)
        }
        assert after == before
        assert db.query(Document).count() == 3
        assert db.query(Chunk).count() == 3

    def test_deleting_the_latest_version_restores_its_parent(self, client, db, chain) -> None:
        _, v2, v3 = chain
        r = client.delete(f"/api/v1/documents/{v3.doc_id}")
        assert r.status_code == 200, r.text
        assert "v2.0 is the latest version again" in r.json()["lineage_message"]

        db.refresh(v2)
        assert v2.is_latest is True
        c2 = chunk_of(db, v2)
        assert (c2.valid_to, c2.is_superseded) == (None, False)

    def test_the_older_versions_are_untouched(self, client, db, chain) -> None:
        # Deleting v3 must restore v2 only. v1's window closed when v2 was
        # published and has nothing to do with v3.
        v1, v2, v3 = chain
        client.delete(f"/api/v1/documents/{v3.doc_id}")

        db.refresh(v1)
        c1 = chunk_of(db, v1)
        assert v1.is_latest is False
        assert (c1.valid_to.replace(tzinfo=None), c1.is_superseded) == (datetime(2023, 3, 15), True)
        assert v2.parent_doc_id == v1.doc_id

    def test_the_chain_can_be_deleted_from_the_top_down(self, client, db, chain) -> None:
        # Every state in between stays valid: exactly one latest version, and
        # every parent link pointing at a document that still exists.
        v1, v2, v3 = chain
        for doomed in (v3, v2, v1):
            assert client.delete(f"/api/v1/documents/{doomed.doc_id}").status_code == 200
            db.expire_all()
            remaining = db.query(Document).all()
            assert [d.is_latest for d in remaining].count(True) == len(remaining[:1])
            ids = {d.doc_id for d in remaining}
            assert all(d.parent_doc_id in ids for d in remaining if d.parent_doc_id)

        assert db.query(Document).count() == 0
        assert db.query(Chunk).count() == 0

    def test_a_new_version_can_be_ingested_after_the_latest_was_deleted(self, client, db, chain) -> None:
        # The restored parent must be a usable parent again, or the delete
        # left the lineage in a state ingestion can't extend.
        _, v2, v3 = chain
        client.delete(f"/api/v1/documents/{v3.doc_id}")
        db.expire_all()

        v3b = ingest(db, "3.1", utc(2025, 6, 1), parent=v2)
        db.refresh(v2)
        assert (v2.is_latest, v3b.is_latest) == (False, True)
        assert chunk_of(db, v2).valid_to.replace(tzinfo=None) == datetime(2025, 6, 1)


class TestIndexBehaviour:
    def test_every_vector_of_a_deleted_document_leaves_faiss(self, client, db) -> None:
        # A document with three chunks holds three FAISS ids; deleting it must
        # take all three, and only those three.
        doomed = ingest(db, "1.0", utc(2024, 1, 1), chunks=3)
        other = ingest(db, "1.0", utc(2024, 2, 1), chunks=2)
        doomed_ids, other_ids = faiss_ids_of(db, doomed), faiss_ids_of(db, other)
        assert len(doomed_ids) == 3
        assert faiss_store.index_size() == 5

        r = client.delete(f"/api/v1/documents/{doomed.doc_id}")
        assert r.status_code == 200, r.text

        assert faiss_store.index_size() == 2
        found = searchable_faiss_ids()
        assert found & set(doomed_ids) == set()      # gone, not merely ranked low
        assert set(other_ids) <= found               # the other document is untouched

    def test_the_chunks_leave_bm25_and_postgres_too(self, client, db) -> None:
        doomed = ingest(db, "1.0", utc(2024, 1, 1), chunks=3)
        ingest(db, "1.0", utc(2024, 2, 1), chunks=2)

        assert bm25_store.corpus_size() == 5
        client.delete(f"/api/v1/documents/{doomed.doc_id}")

        assert bm25_store.corpus_size() == 2
        assert db.query(Chunk).filter(Chunk.doc_id == doomed.doc_id).count() == 0
        assert db.query(Document).filter(Document.doc_id == doomed.doc_id).first() is None

    def test_a_rejected_deletion_leaves_the_indexes_alone(self, client, db, chain) -> None:
        v1, _, _ = chain
        before = (faiss_store.index_size(), bm25_store.corpus_size())

        assert client.delete(f"/api/v1/documents/{v1.doc_id}").status_code == 409

        assert (faiss_store.index_size(), bm25_store.corpus_size()) == before
        assert set(faiss_ids_of(db, v1)) <= searchable_faiss_ids()

    def test_deleting_a_version_removes_only_its_own_vectors(self, client, db) -> None:
        # The lineage case: v3 goes, v1 and v2 stay searchable, and the
        # restored v2 can still be extended by a new version.
        v1 = ingest(db, "1.0", utc(2022, 1, 10), chunks=2)
        v2 = ingest(db, "2.0", utc(2023, 3, 15), v1, chunks=2)
        v3 = ingest(db, "3.0", utc(2025, 2, 1), v2, chunks=2)
        v3_ids, v2_ids, v1_ids = faiss_ids_of(db, v3), faiss_ids_of(db, v2), faiss_ids_of(db, v1)

        assert client.delete(f"/api/v1/documents/{v3.doc_id}").status_code == 200

        found = searchable_faiss_ids()
        assert found & set(v3_ids) == set()
        assert set(v2_ids + v1_ids) <= found

        db.expire_all()
        v4 = ingest(db, "4.0", utc(2025, 8, 1), v2, chunks=1)
        assert set(faiss_ids_of(db, v4)) <= searchable_faiss_ids()

    def test_the_deletion_survives_an_index_reload(self, client, db) -> None:
        # The delete route persists both indexes, so a restart must not bring
        # the vectors back.
        doomed = ingest(db, "1.0", utc(2024, 1, 1), chunks=2)
        keeper = ingest(db, "1.0", utc(2024, 2, 1), chunks=1)
        doomed_ids, keeper_ids = faiss_ids_of(db, doomed), faiss_ids_of(db, keeper)

        client.delete(f"/api/v1/documents/{doomed.doc_id}")

        faiss_store.reset()
        bm25_store.reset()
        faiss_store.load()
        bm25_store.load()

        assert faiss_store.index_size() == 1
        assert bm25_store.corpus_size() == 1
        found = searchable_faiss_ids()
        assert found & set(doomed_ids) == set()
        assert set(keeper_ids) <= found
