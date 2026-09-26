"""
The temporal reranker decides which chunks may answer a question (a hard filter
on each chunk's validity window, chosen by the question's intent) and ranks the
survivors by relevance alone. These tests pin both halves against the running
CEO example from conftest.py:

    v1.0  2022-01-10 → 2023-03-15   Rahul
    v2.0  2023-03-15 → 2025-02-01   Rahul resigned; Priya became CEO
    v3.0  2025-02-01 → (current)    Priya continues
"""

from __future__ import annotations

from datetime import date

import pytest

from conftest import (
    CURRENT,
    HISTORICAL,
    NOW,
    analysis,
    ceo_versions,
    days_before,
    make_chunk,
    pinned,
)
from services.temporal_reranker import (
    describe_filter,
    describe_time_frame,
    rerank,
)


def ids(chunks) -> set[str]:
    return {c.chunk_id for c in chunks}


def in_2023():
    return analysis("range", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))


class TestFilterByIntent:
    def test_current_question_keeps_only_the_current_version(self) -> None:
        assert ids(rerank(ceo_versions(), CURRENT, now=NOW)) == {"v3"}

    def test_who_was_ceo_in_2023_keeps_both_versions_valid_during_2023(self) -> None:
        # Rahul was CEO until 2023-03-15 and Priya after: both are correct for
        # part of 2023. v3.0 only became valid in 2025, so it is excluded.
        assert ids(rerank(ceo_versions(), in_2023(), now=NOW)) == {"v1", "v2"}

    @pytest.mark.parametrize("as_of, expected", [
        (date(2022, 6, 1), "v1"),
        (date(2023, 3, 14), "v1"),    # last day of v1.0's window
        (date(2023, 3, 15), "v2"),    # v2.0 published: the window is [from, to)
        (date(2024, 1, 1), "v2"),
        (date(2025, 2, 1), "v3"),
        (date(2025, 12, 31), "v3"),
    ])
    def test_point_in_time_selects_exactly_the_version_valid_that_day(self, as_of, expected) -> None:
        result = rerank(ceo_versions(), analysis("point_in_time", as_of=as_of), now=NOW)
        assert ids(result) == {expected}

    def test_point_in_time_before_the_first_version_keeps_nothing(self) -> None:
        result = rerank(ceo_versions(), analysis("point_in_time", as_of=date(2021, 1, 1)), now=NOW)
        assert result == []

    def test_range_open_at_the_start(self) -> None:
        before_2023 = analysis("range", end_date=date(2022, 12, 31))
        assert ids(rerank(ceo_versions(), before_2023, now=NOW)) == {"v1"}

    def test_range_open_at_the_end(self) -> None:
        since_mid_2024 = analysis("range", start_date=date(2024, 6, 1))
        assert ids(rerank(ceo_versions(), since_mid_2024, now=NOW)) == {"v2", "v3"}

    def test_range_starting_on_a_supersession_date_excludes_the_superseded_version(self) -> None:
        # v1.0 is valid up to but not including 2023-03-15.
        from_switch = analysis("range", start_date=date(2023, 3, 15), end_date=date(2023, 12, 31))
        assert ids(rerank(ceo_versions(), from_switch, now=NOW)) == {"v2"}

    def test_version_pin_keeps_only_that_version(self) -> None:
        assert ids(rerank(ceo_versions(), pinned("2.0"), now=NOW)) == {"v2"}

    def test_version_pin_matches_trailing_zeros(self) -> None:
        assert ids(rerank(ceo_versions(), pinned("1"), now=NOW)) == {"v1"}

    def test_version_pin_with_no_matching_version_keeps_nothing(self) -> None:
        assert rerank(ceo_versions(), pinned("9.0"), now=NOW) == []

    def test_historical_question_keeps_every_version(self) -> None:
        assert ids(rerank(ceo_versions(), HISTORICAL, now=NOW)) == {"v1", "v2", "v3"}

    def test_undated_chunk_is_valid_for_every_intent(self) -> None:
        undated = make_chunk("undated")
        for a in (CURRENT, in_2023(), analysis("point_in_time", as_of=date(2020, 1, 1)), HISTORICAL):
            assert ids(rerank([undated], a, now=NOW)) == {"undated"}

    def test_superseded_flag_excludes_from_current_even_without_valid_to(self) -> None:
        stale = make_chunk("stale", valid_from=days_before(900), is_superseded=True)
        assert rerank([stale], CURRENT, now=NOW) == []

    def test_not_yet_published_chunk_is_not_current(self) -> None:
        future = make_chunk("future", valid_from=NOW.replace(year=2027))
        assert rerank([future], CURRENT, now=NOW) == []


class TestRankingIsRelevanceOnly:
    """
    Time decides eligibility; relevance decides order. Nothing about a chunk's
    age, version or latest-ness may move it up or down the list.
    """

    @pytest.mark.parametrize("a", [
        pytest.param(CURRENT, id="current"),
        pytest.param(analysis("range", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31)), id="range"),
        pytest.param(analysis("point_in_time", as_of=date(2024, 1, 1)), id="point_in_time"),
        pytest.param(analysis("version", version="2.0"), id="version"),
        pytest.param(analysis("historical"), id="historical"),
    ])
    def test_every_intent_ranks_by_relevance(self, a) -> None:
        scores = [c.relevance_score for c in rerank(ceo_versions(), a, now=NOW)]
        assert scores == sorted(scores, reverse=True)

    def test_past_question_is_not_pulled_toward_the_newest_version(self) -> None:
        """
        The failure this module fixes: with recency applied to "who was CEO in
        2023?", the newer v2.0 would outrank the more relevant v1.0 purely for
        being newer. Relevance alone must decide among valid versions.
        """
        chunks = ceo_versions(rrf=(0.030, 0.020, 0.010))
        result = rerank(chunks, in_2023(), now=NOW)
        assert [c.chunk_id for c in result] == ["v1", "v2"]

    def test_current_question_is_not_pulled_toward_the_newest_document(self) -> None:
        """
        Two unrelated documents, both valid today. The older one matches the
        question far better, so it must win: for a "current" question the
        filter has already removed everything that is out of date, and a
        freshness bonus on top would only demote the better answer.
        """
        candidates = [
            make_chunk("relevant_but_older", doc_id="a", rrf_score=0.03,
                       valid_from=days_before(900)),
            make_chunk("fresh_but_weaker", doc_id="b", rrf_score=0.01,
                       valid_from=days_before(5)),
        ]
        result = rerank(candidates, CURRENT, now=NOW)
        assert [c.chunk_id for c in result] == ["relevant_but_older", "fresh_but_weaker"]

    def test_historical_question_ranks_old_versions_by_relevance(self) -> None:
        result = rerank(ceo_versions(), HISTORICAL, now=NOW)
        assert [c.chunk_id for c in result] == ["v1", "v2", "v3"]

    def test_relevance_is_renormalised_after_filtering(self) -> None:
        # The best CURRENT chunk must score 1.0 even though superseded chunks
        # with higher raw RRF were retrieved and then filtered out.
        chunks = ceo_versions() + [
            make_chunk("other", valid_from=days_before(30), rrf_score=0.001),
        ]
        result = rerank(chunks, CURRENT, now=NOW)
        by_id = {c.chunk_id: c for c in result}
        assert by_id["v3"].relevance_score == 1.0
        assert by_id["other"].relevance_score == 0.0

    def test_ordering_is_deterministic_when_scores_tie(self) -> None:
        # Reproducibility matters for evaluation: identical inputs must give an
        # identical order, so ties break on chunk_id rather than arbitrarily.
        def build():
            return [
                make_chunk("zzz", valid_from=days_before(10)),
                make_chunk("aaa", valid_from=days_before(10)),
            ]

        first = [c.chunk_id for c in rerank(build(), CURRENT, now=NOW)]
        second = [c.chunk_id for c in rerank(build(), CURRENT, now=NOW)]
        assert first == second == ["aaa", "zzz"]

    def test_empty_input_returns_empty(self) -> None:
        assert rerank([], CURRENT, now=NOW) == []


class TestDescriptions:
    def test_filter_descriptions(self) -> None:
        today = NOW.date()
        assert describe_filter(CURRENT, today) == "valid today (2026-01-01)"
        assert describe_filter(in_2023(), today) == "valid at any time from 2023-01-01 to 2023-12-31"
        assert describe_filter(pinned("2.0"), today) == "from version 2.0"
        assert describe_filter(HISTORICAL, today) == "all versions (no time filter)"

    def test_time_frame_is_given_only_for_questions_about_the_past(self) -> None:
        assert describe_time_frame(CURRENT) is None
        assert describe_time_frame(in_2023()) == "2023-01-01 to 2023-12-31"
        assert describe_time_frame(analysis("point_in_time", as_of=date(2024, 3, 31))) == "as of 2024-03-31"
