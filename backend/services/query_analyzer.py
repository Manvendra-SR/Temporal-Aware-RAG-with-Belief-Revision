"""
services/query_analyzer.py — LLM-based temporal query understanding.

analyze(query_text) → QueryAnalysis

One LLM call reads the question and returns its temporal intent as JSON. The
JSON is validated against a fixed schema before anything downstream sees it.

Division of labour
------------------
The LLM only *interprets the question*: "when is the user asking about?".
It never sees any chunks and never decides which sources are valid, relevant,
contradictory or preferred — those stay in deterministic code (the temporal
reranker, the retriever, the NLI detector and belief revision). A bad parse
can therefore change which time window is searched, but cannot inject a fact.

Intents
-------
    current        the present state ("who is the CEO?")
    point_in_time  a single moment ("who was CEO as of March 2024?") → as_of
    range          a period ("who was CEO in 2023?")  → start_date / end_date
    version        a named document version ("according to v2.0 …") → version
    historical     the past, with no specific date ("who used to be CEO?")

A question whose answer does not depend on time at all ("what does the leave
policy cover?") is a "current" question: the answer must come from the version
in force today, which is exactly what "current" retrieves.

Failure behaviour
-----------------
If the call cannot be made (no API key, network error, timeout) or its output
fails validation, analyze() returns the default analysis — intent "current",
which is exactly how the pipeline treats a question with no temporal cues —
with `source="default"` and the reason in `error`, so the UI and the query log
can show that the interpretation was not available. It never raises.

Successful results are cached per (query, today's date), so repeated
questions — and evaluation reruns — cost one call and parse identically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from config import settings

log = logging.getLogger(__name__)

# The analyzer is on the critical path of every query, so it gets a much
# tighter timeout than answer generation. Reasoning models spend tokens before
# the JSON, which is why the token cap is well above the size of the output.
ANALYZER_TIMEOUT_S = 10.0
ANALYZER_MAX_TOKENS = 1024
CACHE_SIZE = 512


class TemporalIntent(str, Enum):
    CURRENT = "current"
    POINT_IN_TIME = "point_in_time"
    RANGE = "range"
    VERSION = "version"
    HISTORICAL = "historical"


@dataclass(frozen=True)
class QueryAnalysis:
    """How the question was interpreted, temporally."""

    intent: TemporalIntent = TemporalIntent.CURRENT
    as_of: Optional[date] = None
    """point_in_time only: the moment the question is about."""

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    """range only, both inclusive. Either may be None for an open-ended range
    ("before 2022", "since March 2024")."""

    version: Optional[str] = None
    """version only: normalised ("v2_0" → "2.0")."""

    source: str = "default"
    """"llm" when the LLM produced this, "default" when it fell back."""

    error: Optional[str] = None
    """Why the LLM result is not available, when source == "default"."""


# ── LLM output schema ───────────────────────────────────────────────────────


class _TemporalParse(BaseModel):
    """
    The JSON contract the LLM must satisfy.

    Fields that are irrelevant to the chosen intent are cleared rather than
    rejected: a model that helpfully fills `as_of` on a "current" question has
    still classified the question correctly. Fields the intent *requires* are
    enforced, because without them the intent cannot be acted on.
    """

    model_config = ConfigDict(extra="ignore")

    intent: TemporalIntent
    as_of: Optional[date] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    version: Optional[str] = None

    @field_validator("as_of", "start_date", "end_date", mode="before")
    @classmethod
    def _dates_are_iso_strings(cls, v: Any) -> Any:
        # Pydantic would read a bare number as a Unix timestamp, silently
        # turning "as_of": 2023 into 1970-01-01. Only ISO strings are accepted.
        if v is None or isinstance(v, str):
            return v
        raise ValueError("dates must be ISO 'YYYY-MM-DD' strings")

    @field_validator("version", mode="before")
    @classmethod
    def _normalise_version(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip().lower().replace("_", ".")
        for prefix in ("version", "release"):
            v = v.removeprefix(prefix).strip()
        v = v.lstrip("v").strip()
        if not v or not any(ch.isdigit() for ch in v):
            return None
        return v

    @model_validator(mode="after")
    def _check_intent_fields(self) -> "_TemporalParse":
        intent = self.intent
        if intent is TemporalIntent.POINT_IN_TIME:
            if self.as_of is None:
                raise ValueError("point_in_time requires as_of")
            self.start_date = self.end_date = self.version = None
        elif intent is TemporalIntent.RANGE:
            if self.start_date is None and self.end_date is None:
                raise ValueError("range requires start_date and/or end_date")
            if self.start_date and self.end_date and self.start_date > self.end_date:
                raise ValueError("range has start_date after end_date")
            self.as_of = self.version = None
        elif intent is TemporalIntent.VERSION:
            if self.version is None:
                raise ValueError("version intent requires a version containing a digit")
            self.as_of = self.start_date = self.end_date = None
        else:
            self.as_of = self.start_date = self.end_date = self.version = None
        return self


_SYSTEM_PROMPT = """\
You classify the TEMPORAL intent of a question asked to a document search system.
The documents are versioned: newer versions supersede older ones. You do NOT answer
the question. You only say which point or period in time it is about.

Return ONLY a JSON object with exactly these keys:
{"intent": ..., "as_of": ..., "start_date": ..., "end_date": ..., "version": ...}

intent is one of:
- "current": about the present state. "Who is the CEO?", "What is the latest policy?"
- "point_in_time": about one specific moment. Set as_of.
    "as of March 2024" -> as_of = last day of that month, "on 5 Jan 2023" -> that day.
- "range": about a period. Set start_date and/or end_date (both inclusive).
    "in 2023" -> 2023-01-01 .. 2023-12-31
    "in March 2024" -> 2024-03-01 .. 2024-03-31
    "before 2022" -> start_date null, end_date 2021-12-31
    "since June 2023" -> start_date 2023-06-01, end_date null
- "version": names a specific document version or release ("v2", "version 2.0",
    "release 3.1"). Set version to just the number, e.g. "2.0". A bare year is a
    date, not a version.
- "historical": about the past with no specific date or version.
    "Who used to be CEO?", "What was the original policy?"

A question whose answer does not depend on time ("what does the leave policy
cover?") is "current": it should be answered from what is in force today.

Rules:
- Dates are ISO "YYYY-MM-DD". Resolve relative expressions ("last year",
  "two months ago") against TODAY, which is given in the user message.
- Every key must be present. Keys not used by the chosen intent are null.
"""


def _user_message(query_text: str, today: date) -> str:
    return f"TODAY: {today.isoformat()}\nQUESTION: {query_text}"


# ── Public entry point ──────────────────────────────────────────────────────


def analyze(
    query_text: str,
    *,
    today: Optional[date] = None,
    client: Any = None,
) -> QueryAnalysis:
    """
    Interpret the temporal intent of *query_text* with one LLM call.

    Args:
        query_text: The raw question.
        today:      Reference date for resolving "last year" etc. Defaults to
                    the current UTC date; injectable for tests.
        client:     A Groq-compatible client. When omitted, the shared client
                    from services.llm is used and results are cached. Passing
                    one (tests) bypasses the cache.

    Returns:
        A QueryAnalysis. Never raises — see "Failure behaviour" above.
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return QueryAnalysis()

    today = today or datetime.now(timezone.utc).date()

    try:
        if client is not None:
            return _parse_with_llm(query_text, today, client)
        return _cached_parse(query_text, today)
    except Exception as exc:  # noqa: BLE001 — any failure degrades to default
        log.warning("query_analyzer: falling back to default analysis: %s", exc)
        return QueryAnalysis(error=_short_error(exc))


# lru_cache does not store calls that raise, so failures (e.g. a missing API
# key that is later added) are retried on the next query rather than cached.
@lru_cache(maxsize=CACHE_SIZE)
def _cached_parse(query_text: str, today: date) -> QueryAnalysis:
    from services.llm import get_client
    return _parse_with_llm(query_text, today, get_client())


def _parse_with_llm(query_text: str, today: date, client: Any) -> QueryAnalysis:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(query_text, today)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=ANALYZER_MAX_TOKENS,
        timeout=ANALYZER_TIMEOUT_S,
    )
    raw = response.choices[0].message.content or ""
    analysis = parse_llm_output(raw)
    log.info(
        "query_analyzer: %r → intent=%s as_of=%s range=%s..%s version=%s",
        query_text[:60], analysis.intent.value, analysis.as_of,
        analysis.start_date, analysis.end_date, analysis.version,
    )
    return analysis


def parse_llm_output(raw: str) -> QueryAnalysis:
    """
    Validate the LLM's raw JSON text and convert it to a QueryAnalysis.

    Raises ValueError if the text is not valid JSON or violates the schema.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {raw[:120]!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"LLM returned JSON that is not an object: {raw[:120]!r}")

    try:
        parsed = _TemporalParse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM output failed validation: {exc.errors()[0]['msg']}") from exc

    return QueryAnalysis(
        intent=parsed.intent,
        as_of=parsed.as_of,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        version=parsed.version,
        source="llm",
    )


def _short_error(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= 200 else text[:197] + "..."
