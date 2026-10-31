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
from models import BOTH_VALID, CONTRADICTION, PREFER_SOURCE, VERSION_CHANGE, ConflictPair
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


def decision(resolution_type: str, preferred: str | None = None, *, note: str = "",
             conflict_type: str = CONTRADICTION, a: str = "old", b: str = "new") -> FakeSession:
    """A session holding one reviewed ConflictPair row for the pair (a, b)."""
    row = ConflictPair(
        conflict_id="c1", chunk_id_a=a, chunk_id_b=b, conflict_type=conflict_type,
        nli_score=0.9, is_resolved=True, resolution_type=resolution_type,
        preferred_chunk_id=preferred, resolution_note=note or None,
    )
    return FakeSession({(a, b): row})


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


def contradiction(chunks, session, intent):
    return revise(chunks, [conflict(CONTRADICTION, "old", "new")], session, intent)


class TestPreferSource:
    """The reviewer names the side that is right; dates play no part."""

    @pytest.mark.parametrize("winner, loser", [("new", "old"), ("old", "new")])
    def test_current_question_excludes_exactly_the_other_side(self, winner, loser) -> None:
        result = contradiction(unrelated_pair(20), decision(PREFER_SOURCE, winner), CURRENT)
        assert result.exclude_chunks == [loser]
        assert result.include_chunks == [winner]
        assert result.preferred_chunks == [winner]
        assert result.answer_confidence == "high"
        assert "reviewer" in result.exclusion_reasons[loser].lower()

    def test_reviewer_can_prefer_the_older_source_over_automatic_recency(self) -> None:
        # Left pending, the 400-day gap would make the newer side win; the
        # reviewer's decision replaces that rule.
        result = contradiction(unrelated_pair(400), decision(PREFER_SOURCE, "old"), CURRENT)
        assert result.exclude_chunks == ["new"]

    def test_a_rejected_newer_claim_is_not_called_superseded(self) -> None:
        notice = contradiction(unrelated_pair(400), decision(PREFER_SOURCE, "old"),
                               CURRENT).conflict_notices[0]
        assert "Arjun is the CEO." in notice          # quoted, so the model can explain
        assert "set-aside" in notice
        assert "superseded" not in notice.lower()

    @pytest.mark.parametrize("dates", ["identical", "missing"])
    def test_works_when_dates_cannot_order_the_pair(self, dates) -> None:
        # Regression: temporal_preference picked "the newer" by date, which for
        # tied or missing dates fell back to UUID order — an arbitrary winner.
        chunks = unrelated_pair(0) if dates == "identical" else unrelated_pair(400)
        if dates == "missing":
            for c in chunks:
                c.valid_from = None
        for winner, loser in (("old", "new"), ("new", "old")):
            result = contradiction(chunks, decision(PREFER_SOURCE, winner), CURRENT)
            assert result.exclude_chunks == [loser]

    @pytest.mark.parametrize("a", PAST_INTENTS)
    def test_past_questions_keep_both_and_explain_the_decision(self, a) -> None:
        result = contradiction(unrelated_pair(400), decision(PREFER_SOURCE, "new"), a)
        assert result.exclude_chunks == []
        assert result.preferred_chunks == []
        assert result.answer_confidence == "high"
        notice = result.conflict_notices[0]
        assert "may still describe its own period" in notice
        assert "rely on Press Release" in notice

    def test_reviewer_note_reaches_the_model(self) -> None:
        result = contradiction(unrelated_pair(20),
                               decision(PREFER_SOURCE, "new", note="checked the registry"), CURRENT)
        assert "Reviewer note: checked the registry" in result.conflict_notices[0]


class TestBothValid:
    @pytest.mark.parametrize("a", [pytest.param(CURRENT, id="current"), *PAST_INTENTS])
    def test_keeps_both_at_high_confidence_for_every_intent(self, a) -> None:
        result = contradiction(unrelated_pair(400), decision(BOTH_VALID), a)
        assert result.exclude_chunks == []
        assert result.preferred_chunks == []
        assert result.answer_confidence == "high"
        assert "do not actually contradict" in result.conflict_notices[0]


class TestDecisionsThatDoNotApply:
    """Anything that is not a well-formed decision on a contradiction is ignored."""

    def automatic(self, session):
        # With a 400-day gap the automatic rule excludes "old" at medium.
        result = contradiction(unrelated_pair(400), session, CURRENT)
        assert (result.exclude_chunks, result.answer_confidence) == (["old"], "medium")

    def test_preferred_chunk_outside_the_pair(self) -> None:
        self.automatic(decision(PREFER_SOURCE, "someone-else"))

    def test_prefer_source_without_a_preferred_chunk(self) -> None:
        self.automatic(decision(PREFER_SOURCE, None))

    @pytest.mark.parametrize("legacy", ["temporal_preference", "scope_clarification", "manual"])
    def test_unmigrated_legacy_types(self, legacy) -> None:
        self.automatic(decision(legacy))

    def test_a_decision_stored_on_a_version_change_is_ignored(self) -> None:
        v1, v2, _ = ceo_versions()
        session = decision(PREFER_SOURCE, "v1", conflict_type=VERSION_CHANGE, a="v1", b="v2")
        result = revise([v1, v2], [conflict(VERSION_CHANGE, "v1", "v2")], session, HISTORICAL)
        assert "VERSION CHANGE" in result.conflict_notices[0]
        assert result.exclude_chunks == []
