"""
Score semantics. These values are combined into the reranking formula and
rendered in the UI, so their ranges and meanings are behavioural contracts.
"""

from __future__ import annotations

import pytest

from conftest import make_chunk
from services.context import build_context
from services.retriever import _assign_relevance_scores, cosine_from_sq_l2


class TestCosineConversion:
    """FAISS returns squared L2 over unit vectors, so cos = 1 - d/2."""

    def test_identical_vectors_score_one(self) -> None:
        assert cosine_from_sq_l2(0.0) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        # Unit vectors 90° apart: ||a-b||² = 2, so cos = 0.
        assert cosine_from_sq_l2(2.0) == pytest.approx(0.0)

    def test_typical_distance_lands_in_a_usable_range(self) -> None:
        """
        Regression: the previous formula 1/(1+d) mapped real distances into a
        ~0.40-0.43 band. Weighted at 0.5 that produced ~0.01 of ranking spread
        against the temporal term's 0.3, so relevance was effectively ignored.
        """
        near = cosine_from_sq_l2(0.6)   # a strong match
        far = cosine_from_sq_l2(1.8)    # a weak one
        assert near - far > 0.5

    def test_missing_distance_scores_zero(self) -> None:
        # A BM25-only hit has no dense distance; it must not be handed an
        # arbitrary similarity.
        assert cosine_from_sq_l2(None) == 0.0

    def test_result_is_clamped_to_unit_range(self) -> None:
        assert cosine_from_sq_l2(4.0) == 0.0     # cos = -1 → clamped
        assert cosine_from_sq_l2(-0.1) == 1.0    # numerical noise → clamped

    def test_is_monotonically_decreasing_in_distance(self) -> None:
        assert cosine_from_sq_l2(0.5) > cosine_from_sq_l2(1.0) > cosine_from_sq_l2(1.5)


class TestRelevanceNormalisation:
    def test_best_and_worst_span_the_unit_range(self) -> None:
        candidates = [
            make_chunk("a", rrf_score=0.030),
            make_chunk("b", rrf_score=0.020),
            make_chunk("c", rrf_score=0.010),
        ]
        _assign_relevance_scores(candidates)

        assert candidates[0].relevance_score == pytest.approx(1.0)
        assert candidates[2].relevance_score == pytest.approx(0.0)
        assert 0.0 < candidates[1].relevance_score < 1.0

    def test_relative_ordering_is_preserved(self) -> None:
        candidates = [
            make_chunk("a", rrf_score=0.031),
            make_chunk("b", rrf_score=0.015),
        ]
        _assign_relevance_scores(candidates)
        assert candidates[0].relevance_score > candidates[1].relevance_score

    def test_all_tied_candidates_score_one(self) -> None:
        candidates = [make_chunk("a", rrf_score=0.02), make_chunk("b", rrf_score=0.02)]
        _assign_relevance_scores(candidates)
        assert all(c.relevance_score == 1.0 for c in candidates)

    def test_single_candidate_scores_one(self) -> None:
        candidates = [make_chunk("a", rrf_score=0.02)]
        _assign_relevance_scores(candidates)
        assert candidates[0].relevance_score == 1.0

    def test_empty_list_is_handled(self) -> None:
        _assign_relevance_scores([])  # must not raise


class TestContextBuilder:
    def test_reports_which_chunks_actually_fitted(self) -> None:
        # The endpoint returns far more candidates than fit in the budget; the
        # UI can only label its Sources list honestly if it knows which ones
        # the model actually saw.
        chunks = [
            make_chunk("a", content="alpha " * 200),
            make_chunk("b", content="beta " * 200),
            make_chunk("c", content="gamma " * 200),
        ]
        context, used = build_context(chunks, budget=250)

        assert used, "at least one chunk should fit"
        assert len(used) < len(chunks), "budget should have truncated the list"
        assert all(cid in {"a", "b", "c"} for cid in used)
        assert "alpha" in context

    def test_budget_is_respected_in_order(self) -> None:
        chunks = [make_chunk("a", content="alpha " * 200), make_chunk("b", content="beta " * 200)]
        _, used = build_context(chunks, budget=250)
        assert used[0] == "a"

    def test_empty_input_produces_empty_output(self) -> None:
        context, used = build_context([], budget=1000)
        assert context == ""
        assert used == []
