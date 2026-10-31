"""
migrations.upgrade() against a database in the pre-redesign state: the
conflict_pairs table without preferred_chunk_id, holding rows written under
the old resolution types and the old date-gap classification.

Each legacy row must end up either as an explicit, applicable decision or back
in the review queue — never as a guessed winner — and running the upgrade
again must change nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import migrations
import models  # noqa: F401 — registers the tables on Base
from database import Base
from models import BOTH_VALID, CONTRADICTION, PREFER_SOURCE, VERSION_CHANGE, Chunk, ConflictPair, Document
from services.conflict_detector import _canonical
from services.version_resolver import apply_lineage


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def doc(db, title: str, published, parent: Document | None = None, *, chunks: int = 1) -> list[Chunk]:
    d = Document(title=title, version_string=f"{1 if parent is None else 2}.0", published_at=published)
    db.add(d)
    db.flush()
    made = [Chunk(doc_id=d.doc_id, chunk_index=i, content=f"{title} {i}", valid_from=published)
            for i in range(chunks)]
    db.add_all(made)
    db.flush()
    d.parent_doc_id = apply_lineage(parent.doc_id if parent else None, d.doc_id,
                                    d.version_string, published, db).parent_doc_id
    return made


@pytest.fixture
def legacy(tmp_path):
    """
    A file-backed SQLite database in the old schema with one legacy row per
    case. (File-backed so upgrade() can open its own connections to it.)
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    (h1, h1b) = doc(db, "Handbook", utc(2022, 1, 10), chunks=2)
    (h2,) = doc(db, "Handbook", utc(2023, 3, 15), parent=h1.document)
    (later,) = doc(db, "Press", utc(2025, 6, 1))
    (same_day,) = doc(db, "Memo", utc(2023, 3, 15))
    (undated,) = doc(db, "Wiki", None)

    def row(name, x, y, conflict_type, resolution_type=None, note=None):
        a, b = _canonical(x.chunk_id, y.chunk_id)
        r = ConflictPair(chunk_id_a=a, chunk_id_b=b, conflict_type=conflict_type, nli_score=0.9,
                         is_resolved=resolution_type is not None, resolution_type=resolution_type,
                         resolution_note=note,
                         resolved_at=utc(2026, 1, 1) if resolution_type else None)
        db.add(r)
        db.flush()
        return name, r.conflict_id

    ids = dict([
        row("dated_preference", h2, later, CONTRADICTION, "temporal_preference", "newer wins"),
        row("same_day_preference", h2, same_day, CONTRADICTION, "temporal_preference", "same day"),
        row("undated_preference", h2, undated, CONTRADICTION, "temporal_preference"),
        row("scope", h1, later, CONTRADICTION, "scope_clarification", "different regions"),
        row("manual", h1, same_day, CONTRADICTION, "manual", "checked upstream"),
        row("resolved_version_change", h1, h2, VERSION_CHANGE, "temporal_preference"),
        row("same_document", h1, h1b, CONTRADICTION),
        row("mistyped_version_change", h1b, h2, CONTRADICTION),
        row("mistyped_contradiction", later, undated, VERSION_CHANGE),
    ])
    expect_newer = later.chunk_id
    db.commit()
    db.close()

    with engine.begin() as conn:   # back to the schema before the redesign
        conn.execute(text("ALTER TABLE conflict_pairs DROP COLUMN preferred_chunk_id"))

    return engine, ids, expect_newer


def load(engine, ids):
    db = sessionmaker(bind=engine)()
    rows = {name: db.get(ConflictPair, cid) for name, cid in ids.items()}
    return db, rows


class TestUpgrade:
    def test_adds_the_column(self, legacy) -> None:
        engine, _, _ = legacy
        migrations.upgrade(engine)
        columns = {c["name"] for c in inspect(engine).get_columns("conflict_pairs")}
        assert "preferred_chunk_id" in columns

    def test_temporal_preference_with_distinct_dates_names_the_later_chunk(self, legacy) -> None:
        engine, ids, newer = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        r = rows["dated_preference"]
        assert (r.is_resolved, r.resolution_type, r.preferred_chunk_id) == (True, PREFER_SOURCE, newer)
        assert r.resolution_note == "newer wins"
        assert r.has_valid_resolution()

    @pytest.mark.parametrize("name", ["same_day_preference", "undated_preference", "manual"])
    def test_rows_without_a_definite_decision_are_reopened(self, legacy, name) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        r = rows[name]
        assert (r.is_resolved, r.resolution_type, r.preferred_chunk_id, r.resolved_at) == (False, None, None, None)
        assert r.needs_review()

    def test_reopening_keeps_the_reviewers_note(self, legacy) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        assert rows["manual"].resolution_note == "checked upstream"

    def test_scope_clarification_becomes_both_valid(self, legacy) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        r = rows["scope"]
        assert (r.resolution_type, r.preferred_chunk_id, r.resolution_note) == (BOTH_VALID, None, "different regions")
        assert r.has_valid_resolution()

    def test_a_resolution_on_a_version_change_is_cleared(self, legacy) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        r = rows["resolved_version_change"]
        assert (r.is_resolved, r.resolution_type) == (False, None)
        assert not r.needs_review()

    def test_same_document_pairs_are_deleted(self, legacy) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        assert rows["same_document"] is None

    @pytest.mark.parametrize("name, expected", [
        ("mistyped_version_change", VERSION_CHANGE),
        ("mistyped_contradiction", CONTRADICTION),
    ])
    def test_every_row_is_retyped_by_lineage(self, legacy, name, expected) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        _, rows = load(engine, ids)
        assert rows[name].conflict_type == expected

    def test_running_it_again_changes_nothing(self, legacy) -> None:
        engine, ids, _ = legacy
        migrations.upgrade(engine)
        db, rows = load(engine, ids)
        snapshot = {n: (r.conflict_type, r.is_resolved, r.resolution_type, r.preferred_chunk_id)
                    for n, r in rows.items() if r is not None}
        db.close()

        migrations.upgrade(engine)
        db = sessionmaker(bind=engine)()
        assert migrations.upgrade_conflict_pairs(db) == {}
        db.close()
        _, rows = load(engine, ids)
        assert {n: (r.conflict_type, r.is_resolved, r.resolution_type, r.preferred_chunk_id)
                for n, r in rows.items() if r is not None} == snapshot

    def test_an_empty_database_is_a_no_op(self, tmp_path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
        Base.metadata.create_all(engine)
        migrations.upgrade(engine)
        migrations.upgrade(engine)
