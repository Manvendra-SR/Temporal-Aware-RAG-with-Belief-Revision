"""
End-to-end tests of POST /query's temporal flow, with only the external pieces
faked: the search indexes, the NLI model's scores, the analyzer's LLM call and
the answer LLM. Everything the project owns runs for real — over-fetching,
temporal filtering and reranking, truncation, conflict typing by lineage,
belief revision, context building and the baseline path.

The corpus is the running example (conftest.py):

    v1.0  2022-01-10 → 2023-03-15   "Rahul is the CEO"
    v2.0  2023-03-15 → 2025-02-01   "Rahul resigned ... Priya became the CEO"
    v3.0  2025-02-01 → (current)    "Priya continues as the CEO"
    press 2025-06-01 → (current)    "Arjun is the CEO"   (unrelated document)
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from conftest import (
    CURRENT,
    HISTORICAL,
    FakeSession,
    analysis,
    ceo_versions,
    make_chunk,
    pinned,
    press_release,
)
from routers import query as query_router
from routers.query import QueryRequest, query_endpoint
from services import llm
from services.conflict_detector import ConflictResult, _canonical, _classify, _select_pairs
from services.retriever import assign_relevance_scores

# Pairs the fake NLI model scores as contradictory. v2.0 and v3.0 agree (Priya).
CONTRADICTING = {
    frozenset({"v1", "v2"}), frozenset({"v1", "v3"}),
    frozenset({"press", "v1"}), frozenset({"press", "v2"}), frozenset({"press", "v3"}),
}


class Pipeline:
    """Installs the fakes and records what each stage received."""

    def __init__(self, monkeypatch, corpus, query_analysis=CURRENT):
        self.corpus = corpus
        self.retrieve_k: int | None = None
        self.analyzer_calls = 0
        self.generated: dict = {}

        def fake_analyze(_query):
            self.analyzer_calls += 1
            return query_analysis

        def fake_retrieve(query, db, k):
            self.retrieve_k = k
            chunks = sorted(self.corpus(), key=lambda c: -c.rrf_score)[:k]
            assign_relevance_scores(chunks)
            return chunks

        def fake_detect(candidates, db, query_id=None):
            results = []
            for a, b in _select_pairs(candidates):
                score = 0.95 if frozenset({a.chunk_id, b.chunk_id}) in CONTRADICTING else 0.05
                conflict_type = _classify(a, b, score)
                if conflict_type:
                    results.append(ConflictResult(*_canonical(a.chunk_id, b.chunk_id),
                                                  conflict_type, score))
            return results

        def fake_generate(context, query, *, temporal=True):
            self.generated = {"context": context, "query": query, "temporal": temporal}
            return "answer"

        monkeypatch.setattr(query_router, "analyze_query", fake_analyze)
        monkeypatch.setattr(query_router, "hybrid_retrieve", fake_retrieve)
        monkeypatch.setattr(query_router, "detect_conflicts", fake_detect)
        monkeypatch.setattr(query_router, "_write_log", lambda *a, **k: None)
        monkeypatch.setattr(llm, "generate", fake_generate)

    def ask(self, question: str, **request):
        return query_endpoint(QueryRequest(query=question, **request), db=FakeSession())

    @property
    def context(self) -> str:
        return self.generated["context"]


def used_ids(response) -> set[str]:
    return {s.chunk_id for s in response.sources if s.used_in_answer}


def handbook_and_press():
    return ceo_versions() + [press_release()]


# ── The headline question ────────────────────────────────────────────────────


class TestWhoWasCeoIn2023:
    def run(self, monkeypatch):
        p = Pipeline(monkeypatch, ceo_versions,
                     analysis("range", start_date=date(2023, 1, 1), end_date=date(2023, 12, 31)))
        return p, p.ask("Who was the CEO in 2023?")

    def test_both_2023_versions_reach_the_answer_and_the_2025_one_does_not(self, monkeypatch) -> None:
        p, response = self.run(monkeypatch)
        assert used_ids(response) == {"v1", "v2"}
        assert "Rahul is the CEO" in p.context
        assert "Priya became the CEO" in p.context
        assert "Priya continues" not in p.context

    def test_the_version_change_is_explained_not_resolved_by_deletion(self, monkeypatch) -> None:
        p, response = self.run(monkeypatch)
        assert response.conflict_pairs[0].conflict_type == "version_supersession"
        assert all(s.excluded_reason is None for s in response.sources)
        assert response.answer_confidence == "high"
        assert "VERSION CHANGE" in p.context

    def test_llm_is_told_the_time_frame_and_each_validity_window(self, monkeypatch) -> None:
        p, _ = self.run(monkeypatch)
        assert p.context.startswith("QUESTION TIME FRAME: 2023-01-01 to 2023-12-31.")
        assert "valid 2022-01-10 → 2023-03-15" in p.context
        assert "valid 2023-03-15 → 2025-02-01" in p.context
        assert p.generated["temporal"] is True

    def test_ranking_is_by_relevance_not_recency(self, monkeypatch) -> None:
        _, response = self.run(monkeypatch)
        assert [s.chunk_id for s in response.sources] == ["v1", "v2"]
        scores = [s.relevance_score for s in response.sources]
        assert scores == sorted(scores, reverse=True)

    def test_response_reports_what_the_filter_did(self, monkeypatch) -> None:
        _, response = self.run(monkeypatch)
        f = response.temporal_filter
        assert f.rule == "valid at any time from 2023-01-01 to 2023-12-31"
        assert (f.candidates_retrieved, f.candidates_valid, f.candidates_kept) == (3, 2, 2)
        assert response.analysis.intent == "range"
        assert response.analysis.start_date == "2023-01-01"


# ── Other intents ────────────────────────────────────────────────────────────


class TestOtherIntents:
    def test_current_question_answers_from_the_current_version_only(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, ceo_versions, CURRENT)
        response = p.ask("Who is the CEO?")
        assert used_ids(response) == {"v3"}
        assert "Rahul" not in p.context
        assert "TIME FRAME" not in p.context
        assert response.temporal_filter.rule.startswith("valid today")

    @pytest.mark.parametrize("as_of, expected", [
        (date(2022, 6, 1), "v1"),
        (date(2024, 1, 1), "v2"),
        (date(2025, 6, 1), "v3"),
    ])
    def test_point_in_time_uses_the_version_valid_that_day(self, monkeypatch, as_of, expected) -> None:
        p = Pipeline(monkeypatch, ceo_versions, analysis("point_in_time", as_of=as_of))
        response = p.ask(f"Who was CEO on {as_of}?")
        assert used_ids(response) == {expected}
        assert p.context.startswith(f"QUESTION TIME FRAME: as of {as_of.isoformat()}.")

    def test_version_pinned_question_uses_only_that_version(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, ceo_versions, pinned("1.0"))
        response = p.ask("According to version 1.0, who was the CEO?")
        assert used_ids(response) == {"v1"}
        assert response.analysis.version_hint == "1.0"

    def test_historical_question_sees_every_version_labelled_with_its_window(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, ceo_versions, HISTORICAL)
        response = p.ask("Who used to be the CEO?")
        assert used_ids(response) == {"v1", "v2", "v3"}
        assert all(s.excluded_reason is None for s in response.sources)
        assert {c.conflict_type for c in response.conflict_pairs} == {"version_supersession"}

    def test_window_with_no_valid_source_returns_no_answer(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, ceo_versions, analysis("point_in_time", as_of=date(2019, 1, 1)))
        response = p.ask("Who was CEO in January 2019?")
        assert response.sources == []
        assert response.answer is None
        assert response.temporal_filter.candidates_valid == 0
        assert p.generated == {}


# ── Conflicting unrelated documents ──────────────────────────────────────────


class TestConflictingDocuments:
    def test_present_question_prefers_the_clearly_newer_unrelated_source(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, handbook_and_press, CURRENT)
        response = p.ask("Who is the CEO?")
        types = {c.conflict_type for c in response.conflict_pairs}
        assert types == {"direct_contradiction"}
        assert used_ids(response) == {"press"}
        excluded = {s.chunk_id: s.excluded_reason for s in response.sources if s.excluded_reason}
        assert set(excluded) == {"v3"}
        assert response.answer_confidence == "medium"

    def test_past_question_keeps_both_sides_of_an_unrelated_conflict(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, handbook_and_press,
                     analysis("range", start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)))
        response = p.ask("Who was CEO in 2025?")
        assert used_ids(response) == {"v2", "v3", "press"}
        assert all(s.excluded_reason is None for s in response.sources)
        assert response.answer_confidence == "low"


# ── Over-fetching ────────────────────────────────────────────────────────────


class TestOverFetch:
    def corpus(self):
        # Six chunks: the four with the highest RRF are all superseded, so
        # retrieving only max_chunks=2 before filtering would leave nothing.
        stale = [
            make_chunk(f"stale{i}", doc_id=f"old{i}", is_superseded=True,
                       rrf_score=0.03 - i * 0.001, content=f"old text {i}")
            for i in range(4)
        ]
        fresh = [
            make_chunk(f"fresh{i}", doc_id=f"new{i}", rrf_score=0.01 - i * 0.001,
                       content=f"new text {i}")
            for i in range(2)
        ]
        return stale + fresh

    def test_retrieves_three_times_max_chunks_then_truncates_after_filtering(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, self.corpus, CURRENT)
        response = p.ask("anything", max_chunks=2)
        assert p.retrieve_k == 6
        assert [s.chunk_id for s in response.sources] == ["fresh0", "fresh1"]
        f = response.temporal_filter
        assert (f.candidates_retrieved, f.candidates_valid, f.candidates_kept) == (6, 2, 2)

    def test_truncates_to_max_chunks_when_more_are_valid(self, monkeypatch) -> None:
        p = Pipeline(monkeypatch, handbook_and_press, HISTORICAL)
        response = p.ask("anything", max_chunks=2)
        assert len(response.sources) == 2
        assert response.temporal_filter.candidates_valid == 4


# ── The baseline ─────────────────────────────────────────────────────────────


class TestBaseline:
    def run(self, monkeypatch):
        p = Pipeline(monkeypatch, ceo_versions, CURRENT)
        return p, p.ask("Who is the CEO?", no_temporal=True)

    def test_every_version_competes_on_relevance_alone(self, monkeypatch) -> None:
        # The problem the project exists to solve: plain RAG hands the model
        # the stale "Rahul" passage first.
        p, response = self.run(monkeypatch)
        assert [s.chunk_id for s in response.sources] == ["v1", "v2", "v3"]
        assert "Rahul is the CEO" in p.context

    def test_no_temporal_stage_runs(self, monkeypatch) -> None:
        p, response = self.run(monkeypatch)
        assert p.analyzer_calls == 0
        assert p.retrieve_k == 20              # max_chunks, no over-fetch
        assert response.temporal_filter is None
        assert response.conflict_pairs == []
        assert response.analysis.source == "skipped"
        assert response.temporal_pipeline_applied is False

    def test_no_temporal_information_reaches_the_llm(self, monkeypatch) -> None:
        p, _ = self.run(monkeypatch)
        assert p.generated["temporal"] is False
        # Document TEXT may mention years ("resigned in March 2023"); what must
        # not appear is metadata — versions, validity dates, notices.
        assert not re.search(r"\d{4}-\d{2}-\d{2}", p.context)
        for marker in ("v1.0", "v2.0", "v3.0", "valid", "TIME FRAME", "CONFIDENCE", "PREFERRED"):
            assert marker not in p.context
