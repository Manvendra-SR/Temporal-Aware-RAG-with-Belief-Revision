"""
Query analysis decides whether a question is version-pinned or historical,
which in turn decides whether superseded chunks stay eligible for retrieval.
A false positive here silently corrupts ranking on ordinary questions.
"""

from __future__ import annotations

import pytest

from services.query_analyzer import analyze


class TestVersionDetection:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("How does autograd work in v2.2?", "2.2"),
            ("What changed in 1.13.1?", "1.13.1"),
            ("Explain the API in PyTorch 1.x", "1.x"),
            ("Show me version 3 behaviour", "3"),
            ("What is new in release 4.1?", "4.1"),
            ("Compare v1_13 with the current build", "1.13"),
        ],
    )
    def test_detects_real_versions(self, query: str, expected: str) -> None:
        assert analyze(query).version_hint == expected

    @pytest.mark.parametrize(
        "query",
        [
            "What are the top 3 methods?",
            "List 5 examples of custom layers",
            "Give me 10 tips for training",
            "How do I use a 2 layer network?",
        ],
    )
    def test_bare_integers_are_not_versions(self, query: str) -> None:
        """
        Regression: the previous pattern `\\bv?(\\d+...)\\b` matched the "3" in
        "top 3 methods". A spurious hint zeroes the version boost for every
        correctly-dated chunk and unlocks superseded content, so an ordinary
        question was ranked as if it were pinned to a nonexistent version.
        """
        analysis = analyze(query)
        assert analysis.version_hint is None
        assert analysis.wants_historical_sources is False

    def test_first_mentioned_version_wins(self) -> None:
        # Matched by two different patterns; document order must decide.
        analysis = analyze("Compare 1.13 against v2.2")
        assert analysis.version_hint == "1.13"
        assert analysis.version_matches == ["1.13", "2.2"]


class TestHistoricalWording:
    @pytest.mark.parametrize(
        "query",
        [
            "Is torch.autograd.Variable deprecated?",
            "How did this work previously?",
            "What was the legacy approach?",
            "This used to require a manual call, right?",
            "Show me the original version of the API",
        ],
    )
    def test_detects_historical_intent(self, query: str) -> None:
        analysis = analyze(query)
        assert analysis.temporal_qualifier is True
        assert analysis.wants_historical_sources is True

    @pytest.mark.parametrize(
        "query",
        [
            "What threshold should I use?",
            "Is the gold standard dataset included?",
            "Should I hold the lock while training?",
            "How do I use the model in various environments?",
        ],
    )
    def test_substring_false_positives_are_gone(self, query: str) -> None:
        """
        Regression: keywords were matched as bare substrings, so "old" fired on
        "threshold"/"gold"/"hold" and "in v" fired on "in various". Each of
        those wrongly unlocked superseded chunks on a current-information
        question.
        """
        assert analyze(query).temporal_qualifier is False


class TestPlainQueries:
    @pytest.mark.parametrize(
        "query",
        [
            "How does mixed precision training work?",
            "What is the recommended optimizer?",
            "",
            "   ",
        ],
    )
    def test_no_hints_on_ordinary_queries(self, query: str) -> None:
        analysis = analyze(query)
        assert analysis.version_hint is None
        assert analysis.temporal_qualifier is False
        assert analysis.wants_historical_sources is False


class TestWantsHistoricalSources:
    def test_version_pin_alone_unlocks_history(self) -> None:
        # You cannot answer "how did this work in v1.13?" without the chunks
        # that v2.x superseded, even with no historical keyword present.
        analysis = analyze("How does the scheduler behave in v1.13?")
        assert analysis.temporal_qualifier is False
        assert analysis.wants_historical_sources is True

    def test_keyword_alone_unlocks_history(self) -> None:
        analysis = analyze("What was the deprecated approach?")
        assert analysis.version_hint is None
        assert analysis.wants_historical_sources is True
