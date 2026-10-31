"""
Version lineage is what gives every chunk its validity window, so the rules
that keep a lineage a single, time-ordered chain are tested here against a real
(in-memory SQLite) database: supersession stamping, the chain guards in both
the early read-only phase (validate_lineage) and the transactional phase
(apply_lineage), and the lineage lookup the conflict detector relies on.

The guards are asserted on both phases because they are two calls into one
implementation: the early one rejects bad metadata before the file is parsed,
chunked and embedded, and the later one re-checks it under the parent's row
lock so a concurrent ingest cannot slip past.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models  # noqa: F401 — registers the tables on Base
from conftest import CURRENT
from database import Base
from models import Chunk, Document
from services.temporal_reranker import is_temporally_valid
from services.version_resolver import apply_lineage, lineage_roots, validate_lineage


def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def ingest(db, version: str, published: datetime, parent: Document | None = None) -> Document:
    """
    What routers/ingest.py does, in the same order: validate the lineage first,
    then insert the document and a chunk, then apply the lineage mutation.
    """
    validate_lineage(parent.doc_id if parent else None, version, published, db)
    doc = Document(title="Handbook", version_string=version, published_at=published)
    db.add(doc)
    db.flush()
    db.add(Chunk(doc_id=doc.doc_id, chunk_index=0, content=f"text of {version}",
                 valid_from=published))
    db.flush()
    lineage = apply_lineage(parent.doc_id if parent else None, doc.doc_id, version, published, db)
    doc.parent_doc_id = lineage.parent_doc_id
    db.commit()
    return doc


def chunk_of(db, doc: Document) -> Chunk:
    return db.query(Chunk).filter(Chunk.doc_id == doc.doc_id).one()


def naive(d: datetime) -> datetime:
    # SQLite returns naive datetimes; compare on the wall-clock value.
    return d.replace(tzinfo=None)


class TestValidityWindows:
    def test_each_version_is_valid_until_the_next_one_is_published(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        v3 = ingest(db, "3.0", utc(2025, 2, 1), parent=v2)

        c1, c2, c3 = chunk_of(db, v1), chunk_of(db, v2), chunk_of(db, v3)
        assert (naive(c1.valid_to), c1.is_superseded) == (naive(utc(2023, 3, 15)), True)
        assert (naive(c2.valid_to), c2.is_superseded) == (naive(utc(2025, 2, 1)), True)
        assert (c3.valid_to, c3.is_superseded) == (None, False)

    def test_only_the_newest_version_is_latest(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        db.refresh(v1)
        assert (v1.is_latest, v2.is_latest) == (False, True)

    def test_a_new_version_supersedes_only_its_parent(self, db) -> None:
        # v1.0's window already closed when v2.0 arrived, so publishing v3.0
        # must not re-stamp it: resolve() only touches the one version that is
        # still current, and that is why it never walks the whole chain.
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        ingest(db, "3.0", utc(2025, 2, 1), parent=v2)

        assert naive(chunk_of(db, v1).valid_to) == naive(utc(2023, 3, 15))
        assert naive(chunk_of(db, v2).valid_to) == naive(utc(2025, 2, 1))

    def test_every_superseded_chunk_has_its_window_closed(self, db) -> None:
        # The CURRENT filter reads only the window, so supersession must never
        # flag a chunk without also giving it a valid_to.
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        ingest(db, "3.0", utc(2025, 2, 1), parent=v2)
        flagged = db.query(Chunk).filter(Chunk.is_superseded == True).all()  # noqa: E712
        assert len(flagged) == 2
        assert all(c.valid_to is not None for c in flagged)


def current_versions(db, today: date) -> set[str]:
    """Which versions the CURRENT filter keeps on `today`, read from the DB rows."""
    return {
        db.get(Document, c.doc_id).version_string
        for c in db.query(Chunk).all()
        if is_temporally_valid(c, CURRENT, today)
    }


class TestFutureDatedVersion:
    """
    v2.0 is uploaded on 2026-01-01 but dated 2026-02-01. Ingestion supersedes
    v1.0 immediately (flag set, valid_to = 2026-02-01); the CURRENT filter must
    still hand the present to v1.0 until v2.0's date arrives.
    """

    @pytest.fixture
    def pair(self, db):
        v1 = ingest(db, "1.0", utc(2025, 1, 1))
        v2 = ingest(db, "2.0", utc(2026, 2, 1), parent=v1)
        return v1, v2

    def test_v1_is_flagged_but_its_window_runs_to_v2s_date(self, db, pair) -> None:
        v1, _ = pair
        c1 = chunk_of(db, v1)
        assert (naive(c1.valid_to), c1.is_superseded) == (naive(utc(2026, 2, 1)), True)

    def test_v1_answers_current_questions_until_v2s_date(self, db, pair) -> None:
        assert current_versions(db, date(2026, 1, 1)) == {"1.0"}
        assert current_versions(db, date(2026, 1, 31)) == {"1.0"}

    def test_v2_takes_over_on_its_date(self, db, pair) -> None:
        assert current_versions(db, date(2026, 2, 1)) == {"2.0"}
        assert current_versions(db, date(2027, 1, 1)) == {"2.0"}


class TestChainGuards:
    def test_new_version_must_extend_the_latest_version(self, db) -> None:
        # Branching from v1.0 after v2.0 exists would leave two "latest"
        # versions and two overlapping validity windows in one lineage.
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        db.refresh(v1)
        with pytest.raises(ValueError, match="not the latest version"):
            ingest(db, "3.0", utc(2025, 2, 1), parent=v1)

    def test_new_version_cannot_predate_its_parent(self, db) -> None:
        # The parent's window would end before it began.
        v1 = ingest(db, "1.0", utc(2023, 1, 1))
        with pytest.raises(ValueError, match="earlier than the parent"):
            ingest(db, "2.0", utc(2022, 6, 1), parent=v1)

    def test_same_day_new_version_is_allowed(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2023, 1, 1))
        ingest(db, "1.1", utc(2023, 1, 1), parent=v1)

    def test_version_must_increase(self, db) -> None:
        v1 = ingest(db, "2.0", utc(2023, 1, 1))
        with pytest.raises(ValueError, match="strictly greater"):
            ingest(db, "1.5", utc(2024, 1, 1), parent=v1)


class TestLineageRoots:
    def test_every_version_maps_to_the_root(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        v2 = ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        v3 = ingest(db, "3.0", utc(2025, 2, 1), parent=v2)
        roots = lineage_roots(db, [v3.doc_id, v2.doc_id, v1.doc_id])
        assert roots == {v1.doc_id: v1.doc_id, v2.doc_id: v1.doc_id, v3.doc_id: v1.doc_id}

    def test_unrelated_documents_have_different_roots(self, db) -> None:
        handbook = ingest(db, "1.0", utc(2022, 1, 10))
        press = ingest(db, "1.0", utc(2025, 6, 1))
        roots = lineage_roots(db, [handbook.doc_id, press.doc_id])
        assert roots[handbook.doc_id] != roots[press.doc_id]

    def test_accepts_any_iterable_and_unknown_ids(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        roots = lineage_roots(db, (i for i in [v1.doc_id, "missing"]))
        assert roots == {v1.doc_id: v1.doc_id, "missing": "missing"}


class TestEarlyValidation:
    """
    validate_lineage() must reject exactly what apply_lineage() would, while
    the new document does not exist yet — that is what lets ingestion run it
    before parsing, chunking and embedding.
    """

    def test_unknown_parent_is_rejected(self, db) -> None:
        with pytest.raises(ValueError, match="not found in the database"):
            validate_lineage("00000000-0000-0000-0000-000000000000", "2.0", utc(2024, 1, 1), db)

    def test_non_latest_parent_is_rejected(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        ingest(db, "2.0", utc(2023, 3, 15), parent=v1)
        db.refresh(v1)
        with pytest.raises(ValueError, match="not the latest version"):
            validate_lineage(v1.doc_id, "3.0", utc(2025, 2, 1), db)

    def test_non_increasing_version_is_rejected(self, db) -> None:
        v1 = ingest(db, "2.0", utc(2023, 1, 1))
        with pytest.raises(ValueError, match="strictly greater"):
            validate_lineage(v1.doc_id, "1.5", utc(2024, 1, 1), db)

    def test_earlier_publication_date_is_rejected(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2023, 1, 1))
        with pytest.raises(ValueError, match="earlier than the parent"):
            validate_lineage(v1.doc_id, "2.0", utc(2022, 6, 1), db)

    def test_a_valid_new_version_passes(self, db) -> None:
        v1 = ingest(db, "1.0", utc(2023, 1, 1))
        validate_lineage(v1.doc_id, "2.0", utc(2024, 1, 1), db)

    def test_a_new_lineage_needs_no_parent(self, db) -> None:
        validate_lineage(None, "1.0", utc(2023, 1, 1), db)

    def test_it_writes_nothing(self, db) -> None:
        # The early phase runs before the new document exists, so it must leave
        # the parent exactly as it found it — superseding happens later.
        v1 = ingest(db, "1.0", utc(2023, 1, 1))
        validate_lineage(v1.doc_id, "2.0", utc(2024, 1, 1), db)
        db.refresh(v1)
        assert v1.is_latest is True
        assert chunk_of(db, v1).is_superseded is False

    def test_the_transactional_phase_rejects_what_the_early_phase_missed(self, db) -> None:
        # The TOCTOU case: two uploads validate against the same latest parent,
        # the first commits, and the second must still be refused even though
        # its early check passed.
        v1 = ingest(db, "1.0", utc(2022, 1, 10))
        validate_lineage(v1.doc_id, "3.0", utc(2025, 2, 1), db)   # second uploader, checks early
        ingest(db, "2.0", utc(2023, 3, 15), parent=v1)            # first uploader commits
        db.refresh(v1)

        doc = Document(title="Handbook", version_string="3.0", published_at=utc(2025, 2, 1))
        db.add(doc)
        db.flush()
        with pytest.raises(ValueError, match="not the latest version"):
            apply_lineage(v1.doc_id, doc.doc_id, "3.0", utc(2025, 2, 1), db)
