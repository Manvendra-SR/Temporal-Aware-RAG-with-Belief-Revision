"""
The context builder is where temporal information reaches the LLM — and so
where it must NOT reach the LLM in the baseline. If the baseline context carried
version numbers and dates, a capable model could pick the newest source by
itself, and a baseline-vs-temporal comparison would measure nothing.
"""

from __future__ import annotations

import re

from conftest import FakeSession, HISTORICAL, ceo_versions, make_chunk
from services import llm
from services.belief_revision import revise
from services.conflict_detector import ConflictResult
from services.context import build_context

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def revision_with_version_change(chunks):
    conflict = ConflictResult("v1", "v2", "version_supersession", 0.9)
    return revise(chunks, [conflict], FakeSession(), HISTORICAL)


class TestBaselineLeaksNothingTemporal:
    def test_headers_carry_only_number_and_title(self) -> None:
        context, _ = build_context(ceo_versions(), temporal_annotations=False)
        headers = [line for line in context.splitlines() if line.startswith("[SOURCE")]
        assert headers == [
            "[SOURCE 1 | ABC Handbook]",
            "[SOURCE 2 | ABC Handbook]",
            "[SOURCE 3 | ABC Handbook]",
        ]

    def test_no_dates_versions_notices_or_confidence(self) -> None:
        chunks = ceo_versions()
        context, _ = build_context(
            chunks,
            revision_result=revision_with_version_change(chunks),
            temporal_annotations=False,
            time_frame="2023-01-01 to 2023-12-31",
        )
        assert not DATE_RE.search(context)
        for marker in ("v1.0", "v2.0", "valid", "TIME FRAME", "VERSION CHANGE",
                       "PREFERRED", "DEPRECATED", "CONFIDENCE"):
            assert marker not in context

    def test_content_is_still_included(self) -> None:
        context, used = build_context(ceo_versions(), temporal_annotations=False)
        assert "Rahul is the CEO" in context
        assert used == ["v1", "v2", "v3"]


class TestTemporalAnnotations:
    def test_superseded_source_shows_its_full_validity_window(self) -> None:
        context, _ = build_context(ceo_versions())
        assert "[SOURCE 1 | ABC Handbook | v1.0 | valid 2022-01-10 → 2023-03-15]" in context

    def test_current_source_shows_an_open_window(self) -> None:
        context, _ = build_context(ceo_versions())
        assert "[SOURCE 3 | ABC Handbook | v3.0 | valid from 2025-02-01]" in context

    def test_time_frame_comes_first(self) -> None:
        context, _ = build_context(ceo_versions(), time_frame="2023-01-01 to 2023-12-31")
        assert context.startswith("QUESTION TIME FRAME: 2023-01-01 to 2023-12-31.")

    def test_no_time_frame_line_for_present_questions(self) -> None:
        context, _ = build_context(ceo_versions())
        assert "TIME FRAME" not in context

    def test_version_change_notice_and_confidence_are_included(self) -> None:
        chunks = ceo_versions()[:2]
        context, _ = build_context(chunks, revision_result=revision_with_version_change(chunks))
        assert "VERSION CHANGE" in context
        assert "ANSWER CONFIDENCE: HIGH" in context
        assert "PREFERRED" not in context


class TestTokenBudget:
    def test_reports_which_chunks_actually_fitted(self) -> None:
        # The endpoint returns more candidates than fit in the budget; the UI
        # can only label its Sources list honestly if it knows which ones the
        # model actually saw.
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


class TestSystemPrompts:
    def _capture_system_prompt(self, monkeypatch, temporal: bool) -> str:
        sent: dict = {}

        class Completions:
            def create(self, **kwargs):
                sent.update(kwargs)
                from types import SimpleNamespace
                msg = SimpleNamespace(content="answer")
                return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

        from types import SimpleNamespace
        monkeypatch.setattr(llm, "get_client",
                            lambda: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
        llm.generate("ctx", "Who is the CEO?", temporal=temporal)
        return sent["messages"][0]["content"]

    def test_baseline_prompt_says_nothing_about_time(self, monkeypatch) -> None:
        prompt = self._capture_system_prompt(monkeypatch, temporal=False)
        assert prompt == llm.BASELINE_SYSTEM_PROMPT
        for word in ("temporal", "valid", "version", "date", "TIME FRAME", "PREFERRED"):
            assert word.lower() not in prompt.lower()

    def test_temporal_prompt_explains_validity_windows(self, monkeypatch) -> None:
        prompt = self._capture_system_prompt(monkeypatch, temporal=True)
        assert prompt == llm.SYSTEM_PROMPT
        assert "QUESTION TIME FRAME" in prompt
