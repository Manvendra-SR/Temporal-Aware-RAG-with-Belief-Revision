"""
The query analyzer turns a question into a temporal intent with one LLM call.
The LLM itself is not under test here — a fake client returns canned JSON. What
is tested is everything the project owns: the request it sends, the validation
of what comes back, how each intent maps onto retrieval policy, and that every
failure degrades to the default interpretation instead of breaking the query.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from services import llm
from services import query_analyzer
from services.query_analyzer import (
    QueryAnalysis,
    TemporalIntent,
    analyze,
    parse_llm_output,
)

TODAY = date(2026, 9, 18)


class FakeClient:
    """Stands in for the Groq client: records requests, returns canned content."""

    def __init__(self, content: str | dict | None = None, error: Exception | None = None):
        self._content = json.dumps(content) if isinstance(content, dict) else content
        self._error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def llm_says(**fields) -> FakeClient:
    payload = {"intent": None, "as_of": None, "start_date": None,
               "end_date": None, "version": None}
    payload.update(fields)
    return FakeClient(payload)


class TestIntents:
    def test_current(self) -> None:
        a = analyze("Who is the CEO?", today=TODAY, client=llm_says(intent="current"))
        assert a.intent is TemporalIntent.CURRENT
        assert a.source == "llm"

    def test_point_in_time(self) -> None:
        a = analyze("Who was CEO as of March 2024?", today=TODAY,
                    client=llm_says(intent="point_in_time", as_of="2024-03-31"))
        assert a.intent is TemporalIntent.POINT_IN_TIME
        assert a.as_of == date(2024, 3, 31)

    def test_range_for_a_bare_year(self) -> None:
        a = analyze("Who was the CEO in 2023?", today=TODAY,
                    client=llm_says(intent="range", start_date="2023-01-01",
                                    end_date="2023-12-31"))
        assert a.intent is TemporalIntent.RANGE
        assert (a.start_date, a.end_date) == (date(2023, 1, 1), date(2023, 12, 31))
        assert a.version is None

    def test_open_ended_range(self) -> None:
        a = analyze("Who was CEO before 2022?", today=TODAY,
                    client=llm_says(intent="range", end_date="2021-12-31"))
        assert a.start_date is None
        assert a.end_date == date(2021, 12, 31)

    def test_version(self) -> None:
        a = analyze("According to version 2.0, who was CEO?", today=TODAY,
                    client=llm_says(intent="version", version="2.0"))
        assert a.intent is TemporalIntent.VERSION
        assert a.version == "2.0"

    def test_historical(self) -> None:
        a = analyze("Who used to be CEO?", today=TODAY, client=llm_says(intent="historical"))
        assert a.intent is TemporalIntent.HISTORICAL
        assert a.version is None

    def test_an_unknown_intent_falls_back_to_current(self) -> None:
        # "atemporal" used to be a sixth intent that filtered and ranked exactly
        # like "current". It was removed, so the model naming it is simply an
        # intent this system does not have — and the fallback is "current",
        # which is the behaviour "atemporal" had anyway.
        a = analyze("What does the leave policy cover?", today=TODAY,
                    client=llm_says(intent="atemporal"))
        assert a.intent is TemporalIntent.CURRENT
        assert a.source == "default"


class TestRequest:
    def test_today_is_sent_so_relative_dates_can_be_resolved(self) -> None:
        client = llm_says(intent="current")
        analyze("Who was CEO last year?", today=TODAY, client=client)
        user_message = client.calls[0]["messages"][-1]["content"]
        assert "2026-09-18" in user_message
        assert "Who was CEO last year?" in user_message

    def test_asks_for_deterministic_json(self) -> None:
        client = llm_says(intent="current")
        analyze("Who is the CEO?", today=TODAY, client=client)
        call = client.calls[0]
        assert call["response_format"] == {"type": "json_object"}
        assert call["temperature"] == 0

    def test_blank_query_makes_no_call(self) -> None:
        client = llm_says(intent="current")
        assert analyze("   ", today=TODAY, client=client) == QueryAnalysis()
        assert client.calls == []


class TestValidation:
    @pytest.mark.parametrize("raw_version, expected", [
        ("2.0", "2.0"),
        ("v2.0", "2.0"),
        ("V1_13", "1.13"),
        ("version 3", "3"),
        ("release 3.1", "3.1"),
        (2, "2"),
    ])
    def test_version_is_normalised(self, raw_version, expected) -> None:
        raw = json.dumps({"intent": "version", "version": raw_version})
        assert parse_llm_output(raw).version == expected

    def test_fields_irrelevant_to_the_intent_are_cleared(self) -> None:
        raw = json.dumps({"intent": "current", "as_of": "2024-01-01", "version": "2.0"})
        a = parse_llm_output(raw)
        assert a.as_of is None
        assert a.version is None

    def test_extra_keys_are_ignored(self) -> None:
        raw = json.dumps({"intent": "historical", "reasoning": "uses 'used to'"})
        assert parse_llm_output(raw).intent is TemporalIntent.HISTORICAL

    @pytest.mark.parametrize("payload", [
        pytest.param({"intent": "point_in_time"}, id="point_in_time without as_of"),
        pytest.param({"intent": "range"}, id="range with no bounds"),
        pytest.param({"intent": "range", "start_date": "2024-01-01", "end_date": "2023-01-01"}, id="range that ends before it starts"),
        pytest.param({"intent": "version", "version": "latest"}, id="version with no digits"),
        pytest.param({"intent": "version"}, id="version intent without a version"),
        pytest.param({"intent": "someday"}, id="unknown intent"),
        pytest.param({"intent": "point_in_time", "as_of": "March 2024"}, id="non-ISO date"),
        pytest.param({"intent": "point_in_time", "as_of": 2023}, id="bare number as a date"),
        pytest.param({"as_of": "2024-01-01"}, id="missing intent"),
    ])
    def test_invalid_output_is_rejected(self, payload) -> None:
        with pytest.raises(ValueError):
            parse_llm_output(json.dumps(payload))

    @pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", '"current"'])
    def test_non_object_output_is_rejected(self, raw) -> None:
        with pytest.raises(ValueError):
            parse_llm_output(raw)


class TestFailureFallsBackToDefault:
    """Whatever goes wrong, the query still runs — as a 'current' question."""

    def assert_default(self, a: QueryAnalysis) -> None:
        assert a.intent is TemporalIntent.CURRENT
        assert a.source == "default"
        assert a.error

    def test_invalid_output(self) -> None:
        a = analyze("Who was CEO?", today=TODAY, client=llm_says(intent="point_in_time"))
        self.assert_default(a)
        assert "as_of" in a.error

    def test_malformed_json(self) -> None:
        self.assert_default(analyze("Who was CEO?", today=TODAY, client=FakeClient("{oops")))

    def test_empty_response(self) -> None:
        self.assert_default(analyze("Who was CEO?", today=TODAY, client=FakeClient(None)))

    def test_client_error(self) -> None:
        client = FakeClient(error=TimeoutError("request timed out"))
        a = analyze("Who was CEO?", today=TODAY, client=client)
        self.assert_default(a)
        assert "timed out" in a.error

    def test_missing_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(llm, "_client", None)
        monkeypatch.setattr(llm.settings, "groq_api_key", "")
        query_analyzer._cached_parse.cache_clear()
        a = analyze("Who is the CEO?", today=TODAY)
        self.assert_default(a)
        assert "GROQ_API_KEY" in a.error


class TestCaching:
    def test_successful_results_are_cached_and_failures_are_not(self, monkeypatch) -> None:
        query_analyzer._cached_parse.cache_clear()
        client = llm_says(intent="historical")
        monkeypatch.setattr(llm, "get_client", lambda: client)

        analyze("Who used to be CEO?", today=TODAY)
        analyze("Who used to be CEO?", today=TODAY)
        assert len(client.calls) == 1

        failing = FakeClient(error=RuntimeError("down"))
        monkeypatch.setattr(llm, "get_client", lambda: failing)
        analyze("Who is CFO?", today=TODAY)
        analyze("Who is CFO?", today=TODAY)
        assert len(failing.calls) == 2
        query_analyzer._cached_parse.cache_clear()
