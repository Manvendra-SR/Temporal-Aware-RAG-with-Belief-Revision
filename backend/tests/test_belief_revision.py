"""
Belief revision arbitrates detected conflicts. The invariant under test:
evidence is only ever EXCLUDED for questions about the present. For questions
about the past, the temporal filter has already chosen the time window, so a
disagreement inside it is explained to the LLM, never deleted.

Every branch must also return an action and a confidence that agree.
"""

from __future__ import annotations

from datetime import date

import pytest

from conftest import (
    CURRENT,
    HISTORICAL,
    FakeSession,
    analysis,
    ceo_versions,
    days_before,
    make_chunk,
    pinned,
    press_release,
)
from services.belief_revision import CONTRADICTION_RECENCY_GAP_DAYS, revise
from services.conflict_detector import ConflictResult


def conflict(type_: str, a: str, b: str, score: float = 0.9) -> ConflictResult:
    return ConflictResult(chunk_id_a=a, chunk_id_b=b, conflict_type=type_, nli_score=score)


def unrelated_pair(gap_days: int) -> list:
    """Two chunks from unrelated documents, the second published `gap_days` later."""
    return [
        make_chunk("old", doc_id="doc-a", doc_title="Handbook", version_string="1.0",
                   content="Rahul is the CEO.", valid_from=days_before(gap_days + 10)),
        make_chunk("new", doc_id="doc-b", doc_title="Press Release", version_string="1.0",
                   content="Arjun is the CEO.", valid_from=days_before(10)),
    ]


IN_2023 = analysis("range", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
PAST_INTENTS = [
    pytest.param(IN_2023, id="range"),
    pytest.param(analysis("point_in_time", as_of=date(2024, 1, 1)), id="point_in_time"),
    pytest.param(HISTORICAL, id="historical"),
    pytest.param(pinned("1.0"), id="version"),
]


class StoredResolution:
    """Stand-in for a resolved ConflictPair row."""

    def __init__(self, resolution_type: str, note: str = ""):
        self.resolution_type = resolution_type
        self.resolution_note = note
        self.is_resolved = True


class TestNoConflicts:
    def test_pass_through_when_nothing_conflicts(self) -> None:
        result = revise(unrelated_pair(400), [], FakeSession(), CURRENT)

        assert result.include_chunks == ["old", "new"]
        assert result.exclude_chunks == []
        assert result.belief_revision_applied is False
        assert result.answer_confidence == "none"
        assert result.preferred_chunks == []

    def test_conflict_about_chunks_not_among_candidates_is_ignored(self) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "x", "y")],
                        FakeSession(), CURRENT)
        assert result.belief_revision_applied is False
        assert result.answer_confidence == "none"


class TestVersionChange:
    """Same lineage: the document changed over time. Both sides are kept."""

    def test_who_was_ceo_in_2023_keeps_rahul_and_priya(self) -> None:
        v1, v2, _ = ceo_versions()
        result = revise([v1, v2], [conflict("version_supersession", "v1", "v2")],
                        FakeSession(), IN_2023)

        assert result.exclude_chunks == []
        assert set(result.include_chunks) == {"v1", "v2"}
        assert result.answer_confidence == "high"
        assert result.belief_revision_applied is True

    def test_notice_gives_each_version_its_validity_window(self) -> None:
        v1, v2, _ = ceo_versions()
        result = revise([v1, v2], [conflict("version_supersession", "v1", "v2")],
                        FakeSession(), IN_2023)
        notice = result.conflict_notices[0]
        assert "VERSION CHANGE" in notice
        assert "2022-01-10 → 2023-03-15" in notice
        assert "2023-03-15 → 2025-02-01" in notice

    def test_older_version_is_named_first_even_when_listed_second(self) -> None:
        v1, v2, _ = ceo_versions()
        result = revise([v2, v1], [conflict("version_supersession", "v2", "v1")],
                        FakeSession(), HISTORICAL)
        notice = result.conflict_notices[0]
        assert notice.index("v1.0") < notice.index("v2.0")

    def test_neither_version_is_marked_preferred(self) -> None:
        # Both are right for their own period; a PREFERRED label would tell the
        # model to discard one of them.
        v1, v2, _ = ceo_versions()
        result = revise([v1, v2], [conflict("version_supersession", "v1", "v2")],
                        FakeSession(), HISTORICAL)
        assert result.preferred_chunks == []

    def test_is_handled_the_same_for_a_present_question(self) -> None:
        # The temporal filter normally prevents this pair for present questions;
        # if it does occur, the rule still explains rather than deletes.
        v1, v2, _ = ceo_versions()
        result = revise([v1, v2], [conflict("version_supersession", "v1", "v2")],
                        FakeSession(), CURRENT)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "high"


class TestUnrelatedContradictionAboutThePresent:
    def test_clearly_newer_source_wins_with_medium_confidence(self) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), CURRENT)

        assert result.exclude_chunks == ["old"]
        assert result.include_chunks == ["new"]
        assert result.preferred_chunks == ["new"]
        assert result.answer_confidence == "medium"
        assert "published 400 days later" in result.exclusion_reasons["old"]

    def test_the_excluded_claim_is_quoted_so_the_change_can_be_described(self) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), CURRENT)
        notices = "\n".join(result.conflict_notices)
        assert "Rahul is the CEO." in notices
        assert "superseded" in notices.lower()

    def test_the_quote_is_length_capped(self) -> None:
        chunks = unrelated_pair(400)
        chunks[0].content = "word " * 500
        result = revise(chunks, [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), CURRENT)
        assert "…" in "\n".join(result.conflict_notices)

    def test_order_of_the_pair_does_not_matter(self) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "new", "old")],
                        FakeSession(), CURRENT)
        assert result.exclude_chunks == ["old"]

    @pytest.mark.parametrize("gap", [20, CONTRADICTION_RECENCY_GAP_DAYS])
    def test_close_dates_keep_both_at_low_confidence(self, gap) -> None:
        result = revise(unrelated_pair(gap), [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), CURRENT)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "low"
        assert f"only {gap} days apart" in result.conflict_notices[0]

    def test_just_over_the_gap_prefers_the_newer(self) -> None:
        result = revise(unrelated_pair(CONTRADICTION_RECENCY_GAP_DAYS + 1),
                        [conflict("direct_contradiction", "old", "new")], FakeSession(), CURRENT)
        assert result.exclude_chunks == ["old"]

    def test_an_undated_source_keeps_both(self) -> None:
        chunks = unrelated_pair(400)
        chunks[0].valid_from = None
        result = revise(chunks, [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), CURRENT)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "low"
        assert "undated" in result.conflict_notices[0]

    def test_current_handbook_versus_newer_press_release(self) -> None:
        _, _, v3 = ceo_versions()
        press = press_release()      # 120 days after v3.0
        result = revise([v3, press], [conflict("direct_contradiction", "press", "v3")],
                        FakeSession(), CURRENT)
        assert result.exclude_chunks == ["v3"]
        assert result.preferred_chunks == ["press"]

    def test_omitted_analysis_is_treated_as_the_present(self) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        FakeSession())
        assert result.exclude_chunks == ["old"]


class TestUnrelatedContradictionAboutThePast:
    @pytest.mark.parametrize("a", PAST_INTENTS)
    def test_nothing_is_excluded(self, a) -> None:
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), a)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "low"
        assert "about the past" in result.conflict_notices[0]

    def test_version_pinned_chunk_is_never_dropped(self) -> None:
        """
        Regression: the reranker surfaced the requested v1.0 passage and belief
        revision then dropped it as older, so the answer described the version
        the user did NOT ask about. Under the present-only exclusion rule this
        cannot happen for any version-pinned question.
        """
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        FakeSession(), pinned("1.0"))
        assert "old" in result.include_chunks


class TestConfidenceAggregation:
    def test_worst_confidence_wins(self) -> None:
        v1, v2, _ = ceo_versions()
        a, b = unrelated_pair(20)
        result = revise(
            [v1, v2, a, b],
            [conflict("version_supersession", "v1", "v2"),
             conflict("direct_contradiction", "old", "new")],
            FakeSession(), HISTORICAL,
        )
        assert result.answer_confidence == "low"
        assert len(result.conflict_notices) == 2


class TestStoredResolutions:
    def test_temporal_preference_excludes_the_older_chunk_for_the_present(self) -> None:
        """
        Regression: stored resolutions used to exclude nothing while reporting
        "high" confidence, so resolving a conflict made retrieval strictly
        worse than leaving it pending.
        """
        session = FakeSession({("old", "new"): StoredResolution("temporal_preference")})
        result = revise(unrelated_pair(20), [conflict("direct_contradiction", "old", "new")],
                        session, CURRENT)
        assert result.exclude_chunks == ["old"]
        assert result.preferred_chunks == ["new"]
        assert result.answer_confidence == "high"

    @pytest.mark.parametrize("a", PAST_INTENTS)
    def test_temporal_preference_keeps_both_for_the_past(self, a) -> None:
        session = FakeSession({("old", "new"): StoredResolution("temporal_preference")})
        result = revise(unrelated_pair(20), [conflict("direct_contradiction", "old", "new")],
                        session, a)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "high"

    def test_scope_clarification_keeps_both_at_high_confidence(self) -> None:
        session = FakeSession({("old", "new"): StoredResolution("scope_clarification")})
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        session, CURRENT)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "high"

    def test_manual_resolution_keeps_both_at_medium_confidence(self) -> None:
        # A human adjudicated but recorded no winner, so nothing can be dropped
        # automatically and confidence must not claim otherwise.
        session = FakeSession({("old", "new"): StoredResolution("manual", "checked upstream")})
        result = revise(unrelated_pair(400), [conflict("direct_contradiction", "old", "new")],
                        session, CURRENT)
        assert result.exclude_chunks == []
        assert result.answer_confidence == "medium"
        assert any("checked upstream" in n for n in result.conflict_notices)
