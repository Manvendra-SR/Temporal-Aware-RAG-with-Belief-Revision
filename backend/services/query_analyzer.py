"""
services/query_analyzer.py — Pure-regex query intent extraction.

analyze(query_text) → QueryAnalysis

Detects:
    version_hint         : str | None   e.g. "1.13", "2.0", "1.x"
    temporal_qualifier   : bool         True when user asks about an older version
    domain_hint          : str | None   e.g. "pytorch", "numpy"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Constants ───────────────────────────────────────────────────────────────

# Words/phrases that signal the user wants an *older* version
_TEMPORAL_KEYWORDS: list[str] = [
    "previously",
    "before",
    "old",
    "older",
    "legacy",
    "deprecated",
    "used to",
    "earlier",
    "prior",
    "historic",
    "history",
    "when was",
    "in version",
    "in v",
    "back in",
    "original",
    "first version",
    "original version",
]

# Known domain names for domain_hint detection
_KNOWN_DOMAINS: list[str] = [
    "pytorch",
    "torch",
    "numpy",
    "pandas",
    "scikit",
    "sklearn",
    "tensorflow",
    "keras",
    "huggingface",
    "transformers",
    "fastapi",
    "langchain",
    "llama",
    "openai",
]

# Regex: matches version-like patterns such as:
#   v1.13, 2.0, 1.x, 2.2.1, 1.0.0
_VERSION_RE = re.compile(
    r"\bv?(\d+(?:[._]\d+)*(?:[._]x)?)\b",
    re.IGNORECASE,
)


# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass
class QueryAnalysis:
    """Structured intent extracted from a raw query string."""

    version_hint: str | None = None
    """The first detected version string (normalised, e.g. '1.13', '2.0', '1.x')."""

    temporal_qualifier: bool = False
    """True when the user appears to be asking about an *older* version of a document."""

    domain_hint: str | None = None
    """A known domain/library name found in the query (lowercase)."""

    raw_matches: list[str] = field(default_factory=list)
    """All version strings found in the query (for debugging)."""


# ── Main entry-point ─────────────────────────────────────────────────────────


def analyze(query_text: str) -> QueryAnalysis:
    """
    Analyse *query_text* and return a :class:`QueryAnalysis`.

    This function is deliberately free of LLM calls — it uses only
    compiled regex and substring searches so it adds negligible latency.

    Args:
        query_text: The raw natural-language query string.

    Returns:
        A :class:`QueryAnalysis` dataclass.
    """
    lower = query_text.lower()

    # 1. Version hint ----------------------------------------------------------
    version_matches = _VERSION_RE.findall(query_text)
    # Normalise: replace underscores with dots, strip leading "v"
    normalised = [v.replace("_", ".").lstrip("vV") for v in version_matches]
    version_hint = normalised[0] if normalised else None

    # 2. Temporal qualifier ----------------------------------------------------
    temporal_qualifier = any(kw in lower for kw in _TEMPORAL_KEYWORDS)

    # Edge case: if a version hint is present (e.g. "PyTorch 1.x"), that
    # itself implies the user may want results from that specific (older) era,
    # so we treat it as a temporal qualifier too.
    if version_hint and not temporal_qualifier:
        temporal_qualifier = True

    # 3. Domain hint -----------------------------------------------------------
    domain_hint: str | None = None
    for domain in _KNOWN_DOMAINS:
        if domain in lower:
            domain_hint = domain
            break

    return QueryAnalysis(
        version_hint=version_hint,
        temporal_qualifier=temporal_qualifier,
        domain_hint=domain_hint,
        raw_matches=version_matches,
    )
