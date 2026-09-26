"""
services/temporal_reranker.py — Intent-aware temporal filtering and ranking.

rerank(candidates, query_analysis, now) → list[CandidateChunk]

Two steps, deliberately kept separate:

1. FILTER (hard) — is this chunk allowed to answer the question at all?
   Decided from the chunk's validity window [valid_from, valid_to), which
   version_resolver maintains: valid_from is the document's publication date,
   valid_to is the publication date of the version that superseded it (NULL
   while it is still current). Windows are compared at day granularity,
   because publication dates are days.

       intent          chunk is kept when
       ─────────────   ─────────────────────────────────────────────────────
       current         it is valid today (not superseded, already published)
       point_in_time   its window contains as_of
       range           its window overlaps [start_date, end_date] (inclusive;
                       either bound may be open)
       version         its document version matches the requested version
       historical      always — "how did it used to be?" needs every version

2. RANK (soft) — among the chunks that passed, which is best?

   By `relevance` alone, for every intent.

   Time decides *eligibility*, not rank. Once the filter has run, every
   surviving chunk is equally valid for the question that was asked, so the
   only thing left to separate them is how well they match the question.
   Adding a freshness bonus on top would push a "who was CEO in 2023?" answer
   back toward the newest document — the exact failure this module exists to
   fix — and for a "current" question it cannot help either: the filter has
   already removed every superseded version, so the newest version of each
   lineage is the only one still in the running.

`relevance` is the fused BM25+dense RRF score, min-max normalised to [0, 1].
It is re-normalised after filtering, so the scale reflects the candidates that
are actually competing rather than ones the filter already removed.

The caller over-fetches and truncates afterwards (see routers/query.py): the
filter can remove many candidates, and truncating first would leave too few.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from services.query_analyzer import QueryAnalysis, TemporalIntent
from services.retriever import CandidateChunk, assign_relevance_scores
from services.version_resolver import version_matches

log = logging.getLogger(__name__)


def rerank(
    candidates: list[CandidateChunk],
    query_analysis: QueryAnalysis,
    now: datetime | None = None,
) -> list[CandidateChunk]:
    """
    Filter *candidates* to those temporally valid for the question, then rank them.

    Args:
        candidates:     Output of :func:`~services.retriever.hybrid_retrieve`.
        query_analysis: Output of :func:`~services.query_analyzer.analyze`.
        now:            Reference time for "today". Defaults to the current UTC
                        time; injectable so tests are deterministic.

    Returns:
        The surviving candidates, sorted best-first, with ``relevance_score``
        re-normalised across the survivors. Not truncated — the caller decides
        how many to keep.
    """
    today = _as_utc(now or datetime.now(timezone.utc)).date()

    kept = filter_by_validity(candidates, query_analysis, today)
    assign_relevance_scores(kept)

    # chunk_id is a secondary key so ties produce a stable, reproducible order
    # across runs — a coin-flip ordering would make evaluation irreproducible.
    kept.sort(key=lambda c: (-c.relevance_score, c.chunk_id))

    log.info(
        "rerank: %d → %d candidates (intent=%s, filter=%s)",
        len(candidates), len(kept), query_analysis.intent.value,
        describe_filter(query_analysis, today),
    )
    return kept


# ── Filtering ────────────────────────────────────────────────────────────────


def filter_by_validity(
    candidates: list[CandidateChunk],
    query_analysis: QueryAnalysis,
    today: date,
) -> list[CandidateChunk]:
    """Keep only the candidates that are temporally valid for the question."""
    return [c for c in candidates if is_temporally_valid(c, query_analysis, today)]


def is_temporally_valid(
    chunk: CandidateChunk,
    query_analysis: QueryAnalysis,
    today: date,
) -> bool:
    """Apply the intent's filter rule (see the module docstring) to one chunk."""
    intent = query_analysis.intent

    if intent is TemporalIntent.CURRENT:
        # is_superseded and valid_to are set together by version_resolver;
        # checking the flag as well keeps superseded text out of current
        # answers even for a row whose valid_to was never stamped.
        return not chunk.is_superseded and valid_on(chunk, today)

    if intent is TemporalIntent.POINT_IN_TIME:
        return valid_on(chunk, query_analysis.as_of)

    if intent is TemporalIntent.RANGE:
        return overlaps(chunk, query_analysis.start_date, query_analysis.end_date)

    if intent is TemporalIntent.VERSION:
        return version_matches(chunk.version_string, query_analysis.version)

    # HISTORICAL: every version is eligible.
    return True


def valid_on(chunk: CandidateChunk, day: date | None) -> bool:
    """
    True when *day* falls inside the chunk's window [valid_from, valid_to).

    A missing valid_from means "valid since forever" and a missing valid_to
    means "still valid", so an undated chunk is valid on every day.
    """
    if day is None:
        return True
    start, end = _day(chunk.valid_from), _day(chunk.valid_to)
    if start is not None and day < start:
        return False
    if end is not None and day >= end:
        return False
    return True


def overlaps(chunk: CandidateChunk, start: date | None, end: date | None) -> bool:
    """
    True when the chunk's window [valid_from, valid_to) shares at least one day
    with the inclusive query range [start, end]. None means an open bound.
    """
    valid_from, valid_to = _day(chunk.valid_from), _day(chunk.valid_to)
    if end is not None and valid_from is not None and valid_from > end:
        return False          # became valid only after the range ended
    if start is not None and valid_to is not None and valid_to <= start:
        return False          # stopped being valid before the range began
    return True


def describe_filter(query_analysis: QueryAnalysis, today: date) -> str:
    """Plain-English description of the filter applied, for the API and logs."""
    a = query_analysis
    if a.intent is TemporalIntent.CURRENT:
        return f"valid today ({today.isoformat()})"
    if a.intent is TemporalIntent.POINT_IN_TIME:
        return f"valid on {a.as_of.isoformat()}"
    if a.intent is TemporalIntent.RANGE:
        if a.start_date and a.end_date:
            return f"valid at any time from {a.start_date.isoformat()} to {a.end_date.isoformat()}"
        if a.end_date:
            return f"valid at any time up to {a.end_date.isoformat()}"
        return f"valid at any time since {a.start_date.isoformat()}"
    if a.intent is TemporalIntent.VERSION:
        return f"from version {a.version}"
    return "all versions (no time filter)"


def describe_time_frame(query_analysis: QueryAnalysis) -> str | None:
    """
    The period a past-oriented question is about, phrased for the LLM prompt.
    None for questions about the present, which need no time frame.
    """
    a = query_analysis
    if a.intent is TemporalIntent.POINT_IN_TIME:
        return f"as of {a.as_of.isoformat()}"
    if a.intent is TemporalIntent.RANGE:
        if a.start_date and a.end_date:
            return f"{a.start_date.isoformat()} to {a.end_date.isoformat()}"
        if a.end_date:
            return f"up to {a.end_date.isoformat()}"
        return f"since {a.start_date.isoformat()}"
    if a.intent is TemporalIntent.VERSION:
        return f"version {a.version} of the documents"
    if a.intent is TemporalIntent.HISTORICAL:
        return "the past (sources from every version are included)"
    return None


def _as_utc(d: datetime) -> datetime:
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _day(d: datetime | None) -> date | None:
    """The UTC calendar day of a timestamp."""
    return None if d is None else _as_utc(d).astimezone(timezone.utc).date()
