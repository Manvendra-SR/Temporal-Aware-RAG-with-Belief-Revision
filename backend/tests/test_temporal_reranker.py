"""
The temporal reranker is the project's core differentiator: it decides whether
a stale-but-relevant chunk outranks a fresh one. These tests pin the decay
curve, the version-pinning boost, the superseded pre-filter and the guarantee
that relevance still meaningfully influences the ranking.
"""

from __future__ import annotations

import pytest

from conftest import NOW, days_before, make_chunk
from services.query_analyzer import analyze
from services.temporal_reranker import (
    W_RELEVANCE,
    W_TEMPORAL,
    rerank,
    temporal_weight_for,
    version_boost_for,
)


class TestTemporalDecay:
    def test_content_aged_one_half_life_scores_one_half(self) -> None:
        assert temporal_weight_for(days_before(180), NOW, half_life=180) == pytest.approx(0.5)

    def test_content_aged_two_half_lives_scores_one_quarter(self) -> None:
        assert temporal_weight_for(days_before(360), NOW, half_life=180) == pytest.approx(0.25)

    def test_brand_new_content_scores_one(self) -> None:
        assert temporal_weight_for(NOW, NOW, half_life=180) == pytest.approx(1.0)

    def test_undated_content_is_treated_as_fresh(self) -> None:
        # Undated chunks must not be silently buried.
        assert temporal_weight_for(None, NOW, half_life=180) == 1.0

    def test_future_dates_are_clamped_not_amplified(self) -> None:
        future = NOW.replace(year=NOW.year + 1)
        assert temporal_weight_for(future, NOW, half_life=180) == pytest.approx(1.0)

    def test_naive_datetimes_do_not_raise(self) -> None:
        naive = days_before(90).replace(tzinfo=None)
        assert 0.0 < temporal_weight_for(naive, NOW, half_life=180) < 1.0


class TestVersionBoost:
    def test_no_hint_is_neutral_for_every_chunk(self) -> None:
        assert version_boost_for("2.2", None) == version_boost_for("1.0", None)

    def test_exact_match_scores_full(self) -> None:
        assert version_boost_for("2.2", "2.2") == 1.0

    def test_trailing_zeros_still_match(self) -> None:
        assert version_boost_for("2.0.0", "2.0") == 1.0

    def test_mismatch_scores_zero(self) -> None:
        assert version_boost_for("1.13", "2.2") == 0.0

    def test_wildcard_matches_the_series(self) -> None:
        assert version_boost_for("1.13", "1.x") == 1.0
        assert version_boost_for("1.0", "1.x") == 1.0
        assert version_boost_for("2.2", "1.x") == 0.0

    def test_component_comparison_is_numeric(self) -> None:
        """
        Regression: matching used str.startswith, so the hint "2.1" wrongly
        matched chunk version "2.10" — a user asking about 2.1 was served 2.10.
        """
        assert version_boost_for("2.10", "2.1") == 0.0
        assert version_boost_for("2.1", "2.1") == 1.0

    def test_unversioned_chunk_scores_zero_when_a_version_was_requested(self) -> None:
        assert version_boost_for(None, "2.2") == 0.0


class TestSupersededPreFilter:
    def test_superseded_chunks_are_dropped_by_default(self) -> None:
        candidates = [
            make_chunk("fresh", valid_from=days_before(10), is_superseded=False),
            make_chunk("stale", valid_from=days_before(900), is_superseded=True),
        ]
        result = rerank(candidates, analyze("How does this work?"), now=NOW)
        assert [c.chunk_id for c in result] == ["fresh"]

    def test_superseded_chunks_survive_a_historical_question(self) -> None:
        candidates = [
            make_chunk("fresh", valid_from=days_before(10), is_superseded=False),
            make_chunk("stale", valid_from=days_before(900), is_superseded=True),
        ]
        result = rerank(candidates, analyze("What was the deprecated approach?"), now=NOW)
        assert {c.chunk_id for c in result} == {"fresh", "stale"}

    def test_superseded_chunks_survive_a_version_pinned_question(self) -> None:
        candidates = [
            make_chunk("v2", valid_from=days_before(10), version_string="2.2"),
            make_chunk("v1", valid_from=days_before(900), version_string="1.13",
                       is_superseded=True, is_latest=False),
        ]
        result = rerank(candidates, analyze("How did autograd work in v1.13?"), now=NOW)
        assert {c.chunk_id for c in result} == {"v2", "v1"}


class TestRanking:
    def test_fresher_chunk_wins_when_relevance_ties(self) -> None:
        candidates = [
            make_chunk("old", relevance_score=0.8, valid_from=days_before(900), is_latest=False),
            make_chunk("new", relevance_score=0.8, valid_from=days_before(5)),
        ]
        result = rerank(candidates, analyze("How does this work?"), now=NOW)
        assert result[0].chunk_id == "new"

    def test_relevance_can_still_outrank_freshness(self) -> None:
        """
        The whole point of fixing the semantic-score scale: a far more relevant
        chunk must be able to beat a slightly fresher but barely relevant one.
        With the old compressed score the temporal term always dominated.
        """
        candidates = [
            make_chunk("relevant", relevance_score=1.0, valid_from=days_before(200)),
            make_chunk("fresh_junk", relevance_score=0.0, valid_from=days_before(0)),
        ]
        result = rerank(candidates, analyze("How does this work?"), now=NOW)
        assert result[0].chunk_id == "relevant"

    def test_relevance_carries_more_weight_than_recency(self) -> None:
        # Guards the documented weighting: relevance is weighted highest and
        # must actually be able to express that, i.e. its full range beats the
        # temporal term's full range.
        assert W_RELEVANCE > W_TEMPORAL

    def test_scores_are_populated_on_every_result(self) -> None:
        result = rerank(
            [make_chunk("a", valid_from=days_before(30))],
            analyze("How does this work?"),
            now=NOW,
        )
        chunk = result[0]
        assert chunk.temporal_score is not None
        assert chunk.version_boost is not None
        assert chunk.composite_score is not None
        assert 0.0 <= chunk.composite_score <= 1.0

    def test_ordering_is_deterministic_when_scores_tie(self) -> None:
        # Reproducibility matters for evaluation: identical inputs must give
        # an identical order, so ties break on chunk_id rather than arbitrarily.
        def build():
            return [
                make_chunk("zzz", relevance_score=0.5, valid_from=days_before(10)),
                make_chunk("aaa", relevance_score=0.5, valid_from=days_before(10)),
            ]

        first = [c.chunk_id for c in rerank(build(), analyze("x"), now=NOW)]
        second = [c.chunk_id for c in rerank(build(), analyze("x"), now=NOW)]
        assert first == second == ["aaa", "zzz"]

    def test_empty_input_returns_empty(self) -> None:
        assert rerank([], analyze("anything"), now=NOW) == []
