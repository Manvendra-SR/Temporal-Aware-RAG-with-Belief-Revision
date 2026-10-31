"""
Human review of conflicts, end to end against a real (in-memory SQLite)
database: the Conflicts API, the "needs review" counts the sidebar and library
show, and belief revision reading the stored decision back on later queries.

The corpus:

    Handbook v1.0  2022-01-10 → 2023-03-15   ┐ one lineage
    Handbook v2.0  2023-03-15 → (current)    ┘
    Handbook v1.0  2025-06-01 → (current)      an UNRELATED document that
                                               happens to share the title and
                                               the version number

and two detected pairs: v1.0 ↔ v2.0 of the lineage (a version change) and
lineage v2.0 ↔ the unrelated document (a contradiction).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import models  # noqa: F401 — registers the tables on Base
from conftest import CURRENT, HISTORICAL, analysis, make_chunk, pinned
from database import Base, get_db
from models import CONTRADICTION, VERSION_CHANGE, Chunk, ConflictPair, Document
from routers import analytics as analytics_router
from routers import conflicts as conflicts_router
from services.belief_revision import revise
from services.conflict_detector import ConflictResult, _canonical
from services.version_resolver import apply_lineage, validate_lineage


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(conflicts_router.router, prefix="/api/v1")
    app.include_router(analytics_router.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c


def ingest(db, title: str, version: str, published, parent: Document | None = None) -> Chunk:
    """What routers/ingest.py does to the database, for a one-chunk document."""
    validate_lineage(parent.doc_id if parent else None, version, published, db)
    doc = Document(title=title, version_string=version, published_at=published)
    db.add(doc)
    db.flush()
    chunk = Chunk(doc_id=doc.doc_id, chunk_index=0, content=f"{title} {version} text",
                  content_snippet=f"{title} {version}", valid_from=published)
    db.add(chunk)
    db.flush()
    doc.parent_doc_id = apply_lineage(parent.doc_id if parent else None, doc.doc_id,
                                      version, published, db).parent_doc_id
    db.commit()
    return chunk


def pair(db, x: Chunk, y: Chunk, conflict_type: str) -> ConflictPair:
    a, b = _canonical(x.chunk_id, y.chunk_id)
    row = ConflictPair(chunk_id_a=a, chunk_id_b=b, conflict_type=conflict_type, nli_score=0.93)
    db.add(row)
    db.commit()
    return row


@pytest.fixture
def corpus(db):
    v1 = ingest(db, "Handbook", "1.0", utc(2022, 1, 10))
    v2 = ingest(db, "Handbook", "2.0", utc(2023, 3, 15), parent=v1.document)
    other = ingest(db, "Handbook", "1.0", utc(2025, 6, 1))
    db.refresh(v1)
    return {
        "v1": v1, "v2": v2, "other": other,
        "version_change": pair(db, v1, v2, VERSION_CHANGE),
        "contradiction": pair(db, v2, other, CONTRADICTION),
    }


def resolve(client, row: ConflictPair, **body):
    return client.post(f"/api/v1/conflicts/{row.conflict_id}/resolve", json=body)


def listing(client, **params) -> dict:
    r = client.get("/api/v1/conflicts", params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestListing:
    def test_only_the_contradiction_needs_review(self, client, corpus) -> None:
        data = listing(client)
        assert (data["total_needs_review"], data["total_resolved"],
                data["total_version_changes"]) == (1, 0, 1)
        flags = {c["conflict_id"]: c["needs_review"] for c in data["conflicts"]}
        assert flags == {corpus["contradiction"].conflict_id: True,
                         corpus["version_change"].conflict_id: False}

    @pytest.mark.parametrize("status, expected", [
        ("needs_review", "contradiction"),
        ("version_change", "version_change"),
    ])
    def test_status_filter(self, client, corpus, status, expected) -> None:
        data = listing(client, status=status)
        assert [c["conflict_id"] for c in data["conflicts"]] == [corpus[expected].conflict_id]
        assert data["total"] == 1

    def test_unknown_status_is_rejected(self, client, corpus) -> None:
        assert client.get("/api/v1/conflicts", params={"status": "pending"}).status_code == 422

    def test_a_version_change_is_listed_earlier_version_first(self, client, corpus) -> None:
        (record,) = listing(client, status="version_change")["conflicts"]
        assert (record["chunk_a"]["version_string"], record["chunk_b"]["version_string"]) == ("1.0", "2.0")

    def test_same_title_and_version_do_not_make_documents_related(self, client, corpus) -> None:
        # The unrelated "Handbook v1.0" shares title and version with the
        # lineage's first version; only lineage decides the type.
        (record,) = listing(client, status="needs_review")["conflicts"]
        assert record["conflict_type"] == CONTRADICTION
        assert record["chunk_a"]["doc_id"] != record["chunk_b"]["doc_id"]


class TestResolving:
    def test_prefer_source_records_the_chosen_chunk(self, client, corpus) -> None:
        winner = corpus["other"].chunk_id
        r = resolve(client, corpus["contradiction"], resolution_type="prefer_source",
                    preferred_chunk_id=winner, resolution_note="  registry confirms  ")
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["is_resolved"], body["needs_review"]) == (True, False)
        assert (body["resolution_type"], body["preferred_chunk_id"]) == ("prefer_source", winner)
        assert body["resolution_note"] == "registry confirms"
        assert body["resolved_at"] is not None

        data = listing(client)
        assert (data["total_needs_review"], data["total_resolved"]) == (0, 1)

    def test_both_valid_prefers_neither_side(self, client, corpus) -> None:
        r = resolve(client, corpus["contradiction"], resolution_type="both_valid")
        assert r.status_code == 200, r.text
        assert (r.json()["resolution_type"], r.json()["preferred_chunk_id"]) == ("both_valid", None)

    def test_a_decision_can_be_changed(self, client, corpus) -> None:
        row = corpus["contradiction"]
        resolve(client, row, resolution_type="prefer_source", preferred_chunk_id=corpus["v2"].chunk_id)
        r = resolve(client, row, resolution_type="prefer_source", preferred_chunk_id=corpus["other"].chunk_id)
        assert r.json()["preferred_chunk_id"] == corpus["other"].chunk_id
        r = resolve(client, row, resolution_type="both_valid")
        assert r.json()["preferred_chunk_id"] is None

    @pytest.mark.parametrize("body", [
        pytest.param({"resolution_type": "prefer_source"}, id="no-preferred-chunk"),
        pytest.param({"resolution_type": "prefer_source",
                      "preferred_chunk_id": "dddddddd-0000-0000-0000-00000000000d"}, id="chunk-outside-pair"),
        pytest.param({"resolution_type": "both_valid", "preferred_chunk_id": "x"}, id="both-valid-with-winner"),
        pytest.param({"resolution_type": "temporal_preference"}, id="retired-type"),
        pytest.param({"resolution_type": "manual"}, id="retired-manual"),
    ])
    def test_ambiguous_or_malformed_decisions_are_rejected(self, client, db, corpus, body) -> None:
        row = corpus["contradiction"]
        if body.get("preferred_chunk_id") == "x":
            body = {**body, "preferred_chunk_id": corpus["v2"].chunk_id}
        assert resolve(client, row, **body).status_code == 422
        db.refresh(row)
        assert row.is_resolved is False

    def test_a_version_change_cannot_be_resolved(self, client, db, corpus) -> None:
        row = corpus["version_change"]
        r = resolve(client, row, resolution_type="both_valid")
        assert r.status_code == 409
        db.refresh(row)
        assert row.is_resolved is False

    def test_unknown_conflict_is_a_404(self, client, corpus) -> None:
        r = client.post("/api/v1/conflicts/eeeeeeee-0000-0000-0000-00000000000e/resolve",
                        json={"resolution_type": "both_valid"})
        assert r.status_code == 404


class TestNeedsReviewCounts:
    def test_sidebar_counts_only_contradictions(self, client, corpus) -> None:
        overview = client.get("/api/v1/analytics/overview").json()
        assert (overview["total_conflicts"], overview["unresolved_conflicts"]) == (2, 1)

    def test_a_version_change_does_not_mark_a_document_as_conflicted(self, client, corpus) -> None:
        nodes = {n["doc_id"]: n for n in client.get("/api/v1/analytics/timeline").json()["nodes"]}
        assert nodes[corpus["v1"].doc_id]["status"] == "superseded"
        assert nodes[corpus["v1"].doc_id]["unresolved_conflicts"] == 0
        assert nodes[corpus["other"].doc_id]["status"] == "conflict"

    def test_resolving_clears_the_conflict_status(self, client, corpus) -> None:
        resolve(client, corpus["contradiction"], resolution_type="both_valid")
        assert client.get("/api/v1/analytics/overview").json()["unresolved_conflicts"] == 0


class TestDecisionReachesQueries:
    """A decision made through the API, read back by belief revision from the table."""

    def candidates(self, corpus):
        v2, other = corpus["v2"], corpus["other"]
        return [
            make_chunk(v2.chunk_id, doc_id=v2.doc_id, doc_title="Handbook", version_string="2.0",
                       content="Priya is the CEO.", valid_from=v2.valid_from),
            make_chunk(other.chunk_id, doc_id=other.doc_id, doc_title="Handbook", version_string="1.0",
                       content="Arjun is the CEO.", valid_from=other.valid_from),
        ]

    def conflicts(self, corpus):
        row = corpus["contradiction"]
        return [ConflictResult(row.chunk_id_a, row.chunk_id_b, CONTRADICTION, 0.93)]

    def test_current_question_follows_the_reviewer(self, client, db, corpus) -> None:
        # The automatic rule would prefer the newer unrelated document; the
        # reviewer says the older one is right.
        keep, drop = corpus["v2"].chunk_id, corpus["other"].chunk_id
        resolve(client, corpus["contradiction"], resolution_type="prefer_source", preferred_chunk_id=keep)
        result = revise(self.candidates(corpus), self.conflicts(corpus), db, CURRENT)
        assert result.exclude_chunks == [drop]
        assert result.preferred_chunks == [keep]
        assert result.answer_confidence == "high"

    @pytest.mark.parametrize("intent", [
        pytest.param(HISTORICAL, id="historical"),
        pytest.param(analysis("range", start_date=datetime(2025, 1, 1).date()), id="range"),
        pytest.param(pinned("1.0"), id="version"),
    ])
    def test_past_question_keeps_the_rejected_source(self, client, db, corpus, intent) -> None:
        resolve(client, corpus["contradiction"], resolution_type="prefer_source",
                preferred_chunk_id=corpus["v2"].chunk_id)
        result = revise(self.candidates(corpus), self.conflicts(corpus), db, intent)
        assert result.exclude_chunks == []
        assert "REVIEWED CONFLICT" in result.conflict_notices[0]

    def test_pending_conflict_uses_the_automatic_rule(self, db, corpus) -> None:
        result = revise(self.candidates(corpus), self.conflicts(corpus), db, CURRENT)
        # More than 90 days older than the unrelated document, so it is set aside.
        assert result.exclude_chunks == [corpus["v2"].chunk_id]
        assert result.answer_confidence == "medium"
