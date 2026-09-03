"""
Belief revision decides which side of a contradiction reaches the LLM and how
confident the answer is allowed to be. Every branch must return an action and a
confidence that agree with each other.
"""

from __future__ import annotations

from conftest import FakeSession, days_before, make_chunk
from services.belief_revision import revise
from services.conflict_detector import ConflictResult
from services.query_analyzer import analyze


def conflict(type_: str, a: str = "old", b: str = "new", score: float = 0.9) -> ConflictResult:
    return ConflictResult(chunk_id_a=a, chunk_id_b=b, conflict_type=type_, nli_score=score)


def dated_pair(gap_days: int) -> list:
    """An older and a newer chunk separated by `gap_days`."""
    return [
        make_chunk("old", valid_from=days_before(gap_days + 10), version_string="1.0",
                   is_latest=False, doc_title="Docs"),
        make_chunk("new", valid_from=days_before(10), version_string="2.0",
                   doc_title="Docs"),
    ]


class StoredResolution:
    """Stand-in for a resolved ConflictPair row."""

    def __init__(self, resolution_type: str, note: str = ""):
        self.resolution_type = resolution_type
        self.resolution_note = note
        self.is_resolved = True


class TestNoConflicts:
    def test_pass_through_when_nothing_conflicts(self) -> None:
        candidates = dated_pair(400)
        result = revise(candidates, [], FakeSession())

        assert result.include_chunks == ["old", "new"]
        assert result.exclude_chunks == []
        assert result.belief_revision_applied is False
        assert result.answer_confidence == "none"
        assert result.preferred_chunks == []


class TestVersionSupersession:
    def test_older_chunk_is_excluded_with_high_confidence(self) -> None:
        result = revise(dated_pair(400), [conflict("version_supersession")], FakeSession())

        assert result.exclude_chunks == ["old"]
        assert result.include_chunks == ["new"]
        assert result.answer_confidence == "high"
        assert result.belief_revision_applied is True

    def test_excluded_chunk_carries_a_reason(self) -> None:
        result = revise(dated_pair(400), [conflict("version_supersession")], FakeSession())
        assert "old" in result.exclusion_reasons
        assert "Superseded by" in result.exclusion_reasons["old"]

    def test_surviving_chunk_is_marked_preferred(self) -> None:
        result = revise(dated_pair(400), [conflict("version_supersession")], FakeSession())
        assert result.preferred_chunks == ["new"]

    def test_the_superseded_claim_is_quoted_in_the_notice(self) -> None:
        """
        The excluded chunk is dropped from the context entirely, so its text
        must be quoted in the notice. Without it the model is told an older
        source was overridden but never shown what it said, and cannot answer
        "what changed?" — it reports the old text as missing from its sources.
        """
        candidates = [
            make_chunk("old", content="Buffering is synchronous by default.",
                       valid_from=days_before(410), is_latest=False),
            make_chunk("new", content="Buffering is asynchronous by default.",
                       valid_from=days_before(10)),
        ]
        result = revise(candidates, [conflict("version_supersession")], FakeSession())

        notices = "\n".join(result.conflict_notices)
        assert "Buffering is synchronous by default." in notices
        assert "superseded" in notices.lower()

    def test_the_quote_is_length_capped(self) -> None:
        candidates = [
            make_chunk("old", content="word " * 500, valid_from=days_before(410), is_latest=False),
            make_chunk("new", content="something else", valid_from=days_before(10)),
        ]
        result = revise(candidates, [conflict("version_supersession")], FakeSession())
        assert "…" in "\n".join(result.conflict_notices)


class TestDirectContradiction:
    def test_wide_date_gap_prefers_the_newer_source(self) -> None:
        # 400-day gap: clear temporal ordering, so the newer source wins.
        result = revise(dated_pair(400), [conflict("direct_contradiction")], FakeSession())

        assert result.exclude_chunks == ["old"]
        assert result.answer_confidence == "medium"

    def test_narrow_date_gap_keeps_both_and_lowers_confidence(self) -> None:
        # 20-day gap: ordering is ambiguous, so neither side can be dismissed.
        result = revise(dated_pair(20), [conflict("direct_contradiction")], FakeSession())

        assert result.exclude_chunks == []
        assert set(result.include_chunks) == {"old", "new"}
        assert result.answer_confidence == "low"

    def test_both_chunks_are_preferred_when_neither_is_dropped(self) -> None:
        result = revise(dated_pair(20), [conflict("direct_contradiction")], FakeSession())
        assert set(result.preferred_chunks) == {"old", "new"}


class TestUnknownConflictType:
    def test_unrecognised_type_keeps_both_at_low_confidence(self) -> None:
        # Safe fallback: never silently drop evidence for a rule we don't have.
        result = revise(dated_pair(400), [conflict("something_new")], FakeSession())

        assert result.exclude_chunks == []
        assert result.answer_confidence == "low"


class TestStoredResolutions:
    def test_temporal_preference_excludes_the_older_chunk(self) -> None:
        """
        Regression: stored resolutions used to exclude nothing while reporting
        "high" confidence, so resolving a conflict made retrieval strictly
        worse than leaving it pending AND told the LLM to be confident about
        an unreconciled contradiction.
        """
        session = FakeSession({("old", "new"): StoredResolution("temporal_preference")})
        result = revise(dated_pair(400), [conflict("direct_contradiction")], session)

        assert result.exclude_chunks == ["old"]
        assert result.answer_confidence == "high"

    def test_scope_clarification_keeps_both_at_high_confidence(self) -> None:
        session = FakeSession({("old", "new"): StoredResolution("scope_clarification")})
        result = revise(dated_pair(400), [conflict("direct_contradiction")], session)

        assert result.exclude_chunks == []
        assert result.answer_confidence == "high"

    def test_manual_resolution_keeps_both_at_medium_confidence(self) -> None:
        # A human adjudicated but recorded no winner, so nothing can be dropped
        # automatically and confidence must not claim otherwise.
        session = FakeSession({("old", "new"): StoredResolution("manual", "checked upstream")})
        result = revise(dated_pair(400), [conflict("direct_contradiction")], session)

        assert result.exclude_chunks == []
        assert result.answer_confidence == "medium"
        assert any("checked upstream" in n for n in result.conflict_notices)


class TestVersionPinProtection:
    """
    When the user pins a version, that version's chunks must survive revision.
    Otherwise the reranker surfaces the requested version and belief revision
    immediately drops it, so the answer describes a different version than the
    one that was asked about.
    """

    def pinned_pair(self) -> list:
        return [
            make_chunk("old", valid_from=days_before(410), version_string="1.0",
                       is_latest=False, is_superseded=True, doc_title="Docs"),
            make_chunk("new", valid_from=days_before(10), version_string="2.0",
                       doc_title="Docs"),
        ]

    def test_pinned_version_is_not_excluded(self) -> None:
        result = revise(
            self.pinned_pair(),
            [conflict("version_supersession")],
            FakeSession(),
            analyze("How did buffering work in v1.0?"),
        )
        assert result.exclude_chunks == []
        assert set(result.include_chunks) == {"old", "new"}

    def test_conflict_is_still_reported_when_protected(self) -> None:
        # Protecting the chunk must not hide the disagreement from the reader.
        result = revise(
            self.pinned_pair(),
            [conflict("version_supersession")],
            FakeSession(),
            analyze("How did buffering work in v1.0?"),
        )
        assert result.belief_revision_applied is True
        assert result.conflict_notices

    def test_a_different_pinned_version_does_not_protect(self) -> None:
        result = revise(
            self.pinned_pair(),
            [conflict("version_supersession")],
            FakeSession(),
            analyze("How does buffering work in v2.0?"),
        )
        assert result.exclude_chunks == ["old"]

    def test_vague_historical_wording_does_not_protect(self) -> None:
        # No specific version was named, so normal supersession handling applies.
        result = revise(
            self.pinned_pair(),
            [conflict("version_supersession")],
            FakeSession(),
            analyze("What was the old way to do this?"),
        )
        assert result.exclude_chunks == ["old"]

    def test_omitting_the_analysis_keeps_previous_behaviour(self) -> None:
        result = revise(self.pinned_pair(), [conflict("version_supersession")], FakeSession())
        assert result.exclude_chunks == ["old"]


class TestConfidenceAggregation:
    def test_worst_confidence_across_conflicts_wins(self) -> None:
        candidates = [
            make_chunk("old", valid_from=days_before(410), is_latest=False),
            make_chunk("new", valid_from=days_before(10)),
            make_chunk("other", valid_from=days_before(20)),
        ]
        conflicts = [
            conflict("version_supersession", "old", "new"),   # high
            conflict("direct_contradiction", "new", "other"),  # low (10-day gap)
        ]
        result = revise(candidates, conflicts, FakeSession())

        assert result.answer_confidence == "low"


class TestMissingDates:
    def test_undated_chunks_do_not_crash_the_decision_tree(self) -> None:
        candidates = [
            make_chunk("old", valid_from=None),
            make_chunk("new", valid_from=None),
        ]
        result = revise(candidates, [conflict("direct_contradiction")], FakeSession())

        # A zero gap is ambiguous, so both sides are kept.
        assert result.exclude_chunks == []
        assert result.answer_confidence == "low"
