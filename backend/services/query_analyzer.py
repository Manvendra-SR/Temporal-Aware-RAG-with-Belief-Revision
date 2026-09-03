"""
services/query_analyzer.py — Pure-regex query intent extraction.

analyze(query_text) → QueryAnalysis

Detects two independent things:

    version_hint        str | None   the version the user asked about ("1.13", "2.x")
    temporal_qualifier  bool         the user used historical wording ("deprecated",
                                     "used to", "in older versions")

These are deliberately kept separate. `temporal_qualifier` reflects *only* what
the user's wording says. Whether superseded chunks should be unlocked is a
retrieval policy decision and lives in temporal_reranker, which reads the
derived `wants_historical_sources` property.

No LLM calls — compiled regex only, so this adds negligible latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Historical-intent keywords ──────────────────────────────────────────────
#
# Matched as whole words/phrases. Substring matching was previously used, which
# produced false positives on very common words: "old" fired on "threshold",
# "gold" and "hold", and "in v" fired on "in various". Every entry below is
# anchored with \b on both sides.
_TEMPORAL_PHRASES: list[str] = [
    "previously",
    "before",
    "old",
    "older",
    "oldest",
    "legacy",
    "deprecated",
    "used to",
    "earlier",
    "prior",
    "historic",
    "historical",
    "history",
    "when was",
    "in version",
    "back in",
    "originally",
    "original version",
    "first version",
    "no longer",
    "superseded",
    "obsolete",
]

_TEMPORAL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _TEMPORAL_PHRASES) + r")\b",
    re.IGNORECASE,
)

# ── Version detection ───────────────────────────────────────────────────────
#
# A bare integer is NOT a version. The previous pattern was
#     \bv?(\d+(?:[._]\d+)*(?:[._]x)?)\b
# which matched the "3" in "what are the top 3 methods?" and the "5" in
# "list 5 examples". That single false positive was severe: a spurious
# version_hint zeroes the version boost for every correctly-dated chunk and
# (under the old coupling) unlocked superseded content on ordinary questions.
#
# A version hint now requires one of three unambiguous forms:
#   1. a "v" prefix              v2, v1.13, v2.2.1
#   2. two or more components    2.2, 1.13.1, 1_13
#   3. an explicit keyword       version 2, release 3
# Each form also accepts an "x" wildcard component: v1.x, 1.x, version 2.x
_VERSION_PATTERNS = [
    # v-prefixed: v2, v1.13, v1.x
    re.compile(r"\bv(\d+(?:[._](?:\d+|x))*)\b", re.IGNORECASE),
    # dotted/underscored multi-component: 2.2, 1.13.1, 1.x
    re.compile(r"\b(\d+(?:[._](?:\d+|x))+)\b", re.IGNORECASE),
    # keyword-introduced: "version 2", "release 3.1"
    re.compile(r"\b(?:version|release)\s+v?(\d+(?:[._](?:\d+|x))*)\b", re.IGNORECASE),
]


@dataclass
class QueryAnalysis:
    """Structured intent extracted from a raw query string."""

    version_hint: str | None = None
    """First detected version, normalised (leading 'v' stripped, '_' → '.')."""

    temporal_qualifier: bool = False
    """True when the query uses explicitly historical wording."""

    version_matches: list[str] = field(default_factory=list)
    """All normalised version strings found, in order (debugging / tests)."""

    @property
    def wants_historical_sources(self) -> bool:
        """
        True when superseded chunks should remain eligible for retrieval.

        That is the case when the user either used historical wording, or
        pinned a specific version — you cannot answer "how did this work in
        v1.13?" without access to chunks that v2.x has superseded.
        """
        return self.temporal_qualifier or self.version_hint is not None


def analyze(query_text: str) -> QueryAnalysis:
    """
    Analyse *query_text* and return a :class:`QueryAnalysis`.

    Args:
        query_text: The raw natural-language query string.

    Returns:
        A :class:`QueryAnalysis` dataclass.
    """
    if not query_text or not query_text.strip():
        return QueryAnalysis()

    # 1. Version hints ---------------------------------------------------------
    # Collected with positions and sorted, so that for a query mentioning
    # several versions the hint is the one the user wrote FIRST, regardless of
    # which pattern happened to match it.
    found: list[tuple[int, str]] = []
    for pattern in _VERSION_PATTERNS:
        for m in pattern.finditer(query_text):
            normalised = m.group(1).replace("_", ".").lstrip("vV").lower()
            if normalised:
                found.append((m.start(), normalised))

    found.sort(key=lambda pair: pair[0])
    matches: list[str] = []
    for _, normalised in found:
        if normalised not in matches:
            matches.append(normalised)

    version_hint = matches[0] if matches else None

    # 2. Historical wording ----------------------------------------------------
    temporal_qualifier = bool(_TEMPORAL_RE.search(query_text))

    return QueryAnalysis(
        version_hint=version_hint,
        temporal_qualifier=temporal_qualifier,
        version_matches=matches,
    )
