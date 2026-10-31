"""
Deterministic helpers of the conflict detection pipeline: canonical pair
ordering (which underpins cache correctness and idempotency), the cheap pair
selection gate, and conflict classification.

The NLI model itself is not exercised — it is a downloaded neural network, so
its behaviour belongs to an integration test, not a unit test. The caching
tests replace it with a fake that records every call, against a real
(in-memory SQLite) conflict_pairs table.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models  # noqa: F401 — registers the tables on Base
from conftest import ceo_versions, days_before, make_chunk, press_release
from database import Base
from models import CONTRADICTION, VERSION_CHANGE, ConflictPair
from services import nli
from services.conflict_detector import (
    NLI_THRESHOLD,
    _canonical,
    _classify,
    _lexical_cosine,
    _nli_input,
    _select_pairs,
    _shares_entity,
    detect,
)


class TestCanonicalPairOrdering:
    def test_order_does_not_matter(self) -> None:
        # This is what makes (A,B) and (B,A) hit the same cached row and the
        # same unique-constraint target; without it every query would re-run
        # NLI and insert duplicates.
        assert _canonical("bbb", "aaa") == _canonical("aaa", "bbb")

    def test_smaller_id_comes_first(self) -> None:
        assert _canonical("bbb", "aaa") == ("aaa", "bbb")

    def test_is_idempotent(self) -> None:
        once = _canonical("x", "y")
        assert _canonical(*once) == once


class TestLexicalCosine:
    def test_identical_texts_score_one(self) -> None:
        assert _lexical_cosine("alpha beta gamma", "alpha beta gamma") == pytest.approx(1.0)

    def test_disjoint_texts_score_zero(self) -> None:
        assert _lexical_cosine("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap_is_between(self) -> None:
        score = _lexical_cosine("alpha beta gamma", "alpha beta delta")
        assert 0.0 < score < 1.0

    def test_empty_input_scores_zero(self) -> None:
        assert _lexical_cosine("", "alpha") == 0.0
        assert _lexical_cosine("alpha", "") == 0.0

    def test_is_symmetric(self) -> None:
        a, b = "alpha beta gamma", "beta delta"
        assert _lexical_cosine(a, b) == pytest.approx(_lexical_cosine(b, a))

    def test_stays_within_unit_range(self) -> None:
        # It is a cosine over binary vectors, so it can never exceed 1.0 —
        # which is what makes comparing it against a 0.55 threshold meaningful.
        assert 0.0 <= _lexical_cosine("a b c d e", "a b") <= 1.0


class TestEntitySharing:
    def test_shared_camel_case_identifier_is_detected(self) -> None:
        assert _shares_entity(
            "Use MultiheadAttention for this.",
            "MultiheadAttention was changed.",
        )

    def test_shared_version_number_is_detected(self) -> None:
        assert _shares_entity("Introduced in 2.2 release", "Removed in 2.2")

    def test_unrelated_texts_share_nothing(self) -> None:
        assert not _shares_entity("plain lowercase words here", "different plain words")


class TestPairSelection:
    def test_pairs_sharing_an_entity_are_selected(self) -> None:
        chunks = [
            make_chunk("a", doc_id="d1", content="MultiheadAttention is recommended."),
            make_chunk("b", doc_id="d2", content="MultiheadAttention is deprecated."),
        ]
        assert len(_select_pairs(chunks)) == 1

    def test_unrelated_pairs_are_filtered_out(self) -> None:
        # The gate exists to keep NLI off the full O(n²) pair set.
        chunks = [
            make_chunk("a", doc_id="d1", content="apples grow on trees in orchards"),
            make_chunk("b", doc_id="d2", content="rockets require enormous thrust levels"),
        ]
        assert _select_pairs(chunks) == []

    def test_single_chunk_yields_no_pairs(self) -> None:
        assert _select_pairs([make_chunk("a")]) == []

    def test_same_document_pairs_are_never_selected(self) -> None:
        """
        Two passages of one document are neither a version change nor a
        disagreement between sources, so no rule can act on them. In practice
        NLI flags topically adjacent prose from one long document as
        contradictory, and because confidence is the WORST across pairs, those
        false positives dragged genuinely-resolved answers down to "low" and
        inflated the conflict count.
        """
        chunks = [
            make_chunk("a", doc_id="same-doc", content="MultiheadAttention is recommended."),
            make_chunk("b", doc_id="same-doc", content="MultiheadAttention is deprecated."),
        ]
        assert _select_pairs(chunks) == []

    def test_cross_document_pairs_are_still_selected(self) -> None:
        chunks = [
            make_chunk("a", doc_id="v1", content="MultiheadAttention is recommended."),
            make_chunk("b", doc_id="v2", content="MultiheadAttention is deprecated."),
        ]
        assert len(_select_pairs(chunks)) == 1


class TestClassification:
    """NLI decides whether two chunks conflict; lineage decides the type."""

    def test_below_threshold_is_not_a_conflict(self) -> None:
        v1, v2, _ = ceo_versions()
        assert _classify(v1, v2, NLI_THRESHOLD - 0.01) is None

    def test_threshold_itself_counts_as_a_conflict(self) -> None:
        v1, v2, _ = ceo_versions()
        assert _classify(v1, v2, NLI_THRESHOLD) is not None

    def test_versions_of_one_document_are_version_supersession(self) -> None:
        v1, v2, v3 = ceo_versions()
        assert _classify(v1, v2, 0.9) == "version_supersession"
        assert _classify(v1, v3, 0.9) == "version_supersession"

    def test_unrelated_documents_are_a_direct_contradiction(self) -> None:
        _, _, v3 = ceo_versions()
        assert _classify(v3, press_release(), 0.9) == "direct_contradiction"

    def test_close_publication_dates_do_not_make_versions_a_direct_contradiction(self) -> None:
        a = make_chunk("a", doc_id="d1", lineage_id="L", valid_from=days_before(10))
        b = make_chunk("b", doc_id="d2", lineage_id="L", valid_from=days_before(15))
        assert _classify(a, b, 0.9) == "version_supersession"

    def test_distant_publication_dates_do_not_make_unrelated_documents_versions(self) -> None:
        """
        Regression: the type used to be decided by a 90-day date gap alone, so
        two unrelated documents published a year apart were labelled a
        "version_supersession" and resolved with high confidence.
        """
        a = make_chunk("a", doc_id="d1", valid_from=days_before(0))
        b = make_chunk("b", doc_id="d2", valid_from=days_before(400))
        assert _classify(a, b, 0.9) == "direct_contradiction"

    def test_missing_lineage_falls_back_to_the_document(self) -> None:
        a = make_chunk("a", doc_id="d1")
        b = make_chunk("b", doc_id="d2")
        a.lineage_id = b.lineage_id = ""
        assert _classify(a, b, 0.9) == "direct_contradiction"

    def test_classification_is_symmetric(self) -> None:
        _, _, v3 = ceo_versions()
        press = press_release()
        assert _classify(v3, press, 0.9) == _classify(press, v3, 0.9)


class TestClassificationAcrossLineages:
    """Version numbers are only meaningful within one lineage."""

    def test_identical_versions_of_independent_documents_are_a_contradiction(self) -> None:
        a = make_chunk("a", doc_id="handbook", version_string="1.0", lineage_id="handbook")
        b = make_chunk("b", doc_id="press", version_string="1.0", lineage_id="press")
        assert _classify(a, b, 0.9) == CONTRADICTION

    def test_different_versions_of_independent_documents_are_a_contradiction(self) -> None:
        a = make_chunk("a", doc_id="handbook", version_string="3.0", lineage_id="handbook")
        b = make_chunk("b", doc_id="press", version_string="1.0", lineage_id="press")
        assert _classify(a, b, 0.9) == CONTRADICTION

    @pytest.mark.parametrize("dates", ["identical", "missing"])
    def test_dates_play_no_part_for_versions_of_one_lineage(self, dates) -> None:
        when = days_before(10) if dates == "identical" else None
        a = make_chunk("a", doc_id="v1", lineage_id="L", valid_from=when)
        b = make_chunk("b", doc_id="v2", lineage_id="L", valid_from=when)
        assert _classify(a, b, 0.9) == VERSION_CHANGE


class TestNliInputOrder:
    def test_a_pair_is_scored_in_the_same_order_whichever_way_it_arrives(self) -> None:
        # NLI is directional; the cached score must not depend on retrieval order.
        a = make_chunk("aaa", content="premise")
        b = make_chunk("bbb", content="hypothesis")
        assert _nli_input(a, b) == _nli_input(b, a) == ("premise", "hypothesis")


# ── Caching, against a real table ────────────────────────────────────────────


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


class FakeNli:
    def __init__(self, monkeypatch, score: float = 0.95):
        self.calls: list[list[tuple[str, str]]] = []
        self.score = score
        monkeypatch.setattr(nli, "is_loaded", lambda: True)
        monkeypatch.setattr(nli, "predict", self.predict)

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.score] * len(pairs)


# Hex letters matter: SQLite gives a column declared UUID numeric affinity,
# so an all-digit id would be stored as an integer.
C1 = "aaaaaaaa-0000-0000-0000-00000000000a"
C2 = "bbbbbbbb-0000-0000-0000-00000000000b"


def two_versions():
    return [
        make_chunk(C1, doc_id="d1", lineage_id="L", content="MultiheadAttention is recommended."),
        make_chunk(C2, doc_id="d2", lineage_id="L", content="MultiheadAttention is deprecated."),
    ]


def stored(db, *, conflict_type, score=0.95, **fields) -> ConflictPair:
    row = ConflictPair(conflict_id="cccccccc-0000-0000-0000-00000000000c", chunk_id_a=C1, chunk_id_b=C2,
                       conflict_type=conflict_type, nli_score=score, **fields)
    db.add(row)
    db.commit()
    return row


class TestCaching:
    def test_a_new_pair_is_scored_once_and_then_served_from_the_table(self, db, monkeypatch) -> None:
        fake = FakeNli(monkeypatch)
        first = detect(two_versions(), db)
        second = detect(list(reversed(two_versions())), db)

        assert len(fake.calls) == 1                       # second query ran no NLI
        assert [(r.conflict_type, r.is_cached) for r in first] == [(VERSION_CHANGE, False)]
        assert [(r.conflict_type, r.is_cached) for r in second] == [(VERSION_CHANGE, True)]
        assert db.query(ConflictPair).count() == 1

    def test_a_cached_row_with_a_stale_type_is_retyped_from_lineage(self, db, monkeypatch) -> None:
        # Rows written under the old date-gap rule could call a version change
        # a contradiction; loading the row corrects it, without any NLI call.
        fake = FakeNli(monkeypatch)
        row = stored(db, conflict_type=CONTRADICTION)
        results = detect(two_versions(), db)
        assert fake.calls == []
        assert results[0].conflict_type == VERSION_CHANGE
        db.refresh(row)
        assert row.conflict_type == VERSION_CHANGE

    def test_retyping_leaves_a_reviewers_decision_in_place(self, db, monkeypatch) -> None:
        FakeNli(monkeypatch)
        row = stored(db, conflict_type=CONTRADICTION, is_resolved=True,
                     resolution_type="both_valid", resolution_note="kept")
        detect(two_versions(), db)
        db.refresh(row)
        assert (row.is_resolved, row.resolution_note) == (True, "kept")

    def test_a_cached_score_below_the_threshold_is_not_a_conflict(self, db, monkeypatch) -> None:
        FakeNli(monkeypatch)
        stored(db, conflict_type=VERSION_CHANGE, score=NLI_THRESHOLD - 0.01)
        assert detect(two_versions(), db) == []

    def test_same_document_chunks_never_reach_nli(self, db, monkeypatch) -> None:
        fake = FakeNli(monkeypatch)
        chunks = two_versions()
        chunks[1].doc_id = "d1"
        assert detect(chunks, db) == []
        assert fake.calls == []
