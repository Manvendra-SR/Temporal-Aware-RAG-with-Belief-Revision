"""
Score semantics. These values are combined into the reranking formula and
rendered in the UI, so their ranges and meanings are behavioural contracts.
"""

from __future__ import annotations

import pytest

from conftest import make_chunk
from services.retriever import assign_relevance_scores, clamp_cosine


class TestSemanticScore:
    """
    The FAISS index is IndexFlatIP over unit vectors, so the score it returns
    is already the cosine similarity. clamp_cosine only bounds it.
    """

    def test_identical_vectors_score_one(self) -> None:
        assert clamp_cosine(1.0) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert clamp_cosine(0.0) == pytest.approx(0.0)

    def test_the_score_passes_through_unchanged(self) -> None:
        """
        Regression: an older build scored 1/(1+d) over L2 distances, mapping
        real matches into a ~0.40-0.43 band. Weighted at 0.5 that produced
        ~0.01 of ranking spread against the temporal term's 0.3, so relevance
        was effectively ignored. A cosine spends the whole range.
        """
        near = clamp_cosine(0.70)   # a strong match
        far = clamp_cosine(0.10)    # a weak one
        assert (near, far) == (0.7, 0.1)
        assert near - far > 0.5

    def test_missing_similarity_scores_zero(self) -> None:
        # A BM25-only hit was never returned by the dense index; it must not
        # be handed an invented similarity.
        assert clamp_cosine(None) == 0.0

    def test_result_is_clamped_to_unit_range(self) -> None:
        assert clamp_cosine(-0.8) == 0.0     # unrelated → clamped
        assert clamp_cosine(1.0000001) == 1.0  # float noise → clamped

    def test_is_monotonically_increasing_in_similarity(self) -> None:
        assert clamp_cosine(0.75) > clamp_cosine(0.50) > clamp_cosine(0.25)


class TestRelevanceNormalisation:
    def test_best_and_worst_span_the_unit_range(self) -> None:
        candidates = [
            make_chunk("a", rrf_score=0.030),
            make_chunk("b", rrf_score=0.020),
            make_chunk("c", rrf_score=0.010),
        ]
        assign_relevance_scores(candidates)

        assert candidates[0].relevance_score == pytest.approx(1.0)
        assert candidates[2].relevance_score == pytest.approx(0.0)
        assert 0.0 < candidates[1].relevance_score < 1.0

    def test_relative_ordering_is_preserved(self) -> None:
        candidates = [
            make_chunk("a", rrf_score=0.031),
            make_chunk("b", rrf_score=0.015),
        ]
        assign_relevance_scores(candidates)
        assert candidates[0].relevance_score > candidates[1].relevance_score

    def test_all_tied_candidates_score_one(self) -> None:
        candidates = [make_chunk("a", rrf_score=0.02), make_chunk("b", rrf_score=0.02)]
        assign_relevance_scores(candidates)
        assert all(c.relevance_score == 1.0 for c in candidates)

    def test_single_candidate_scores_one(self) -> None:
        candidates = [make_chunk("a", rrf_score=0.02)]
        assign_relevance_scores(candidates)
        assert candidates[0].relevance_score == 1.0

    def test_empty_list_is_handled(self) -> None:
        assign_relevance_scores([])  # must not raise
