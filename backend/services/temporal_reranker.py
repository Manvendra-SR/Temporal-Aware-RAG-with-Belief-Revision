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
       atemporal       same as current — time does not matter to the question,
                       so there is no reason to serve outdated text
       point_in_time   its window contains as_of
       range           its window overlaps [start_date, end_date] (inclusive;
                       either bound may be open)
       version         its document version matches the requested version
       historical      always — "how did it used to be?" needs every version

2. SCORE (soft) — among the chunks that passed, which is best?

   current:
       composite = 0.5·relevance + 0.3·freshness + 0.1·version_boost + 0.1·latest
       freshness = 2 ** (-age_days / half_life_days)

   every other intent:
       composite = relevance

   Freshness only makes sense when the question is about the present. For
   "who was CEO in 2023?" the filter has already restricted the candidates to
   the right period; adding a recency bonus on top would push the answer back
   toward the newest document — the exact failure this module exists to fix.

`relevance` is the fused BM25+dense RRF score, min-max normalised to [0, 1].
It is re-normalised after filtering, so the scale reflects the candidates that
are actually competing rather than ones the filter already removed.

The caller over-fetches and truncates afterwards (see routers/query.py): the
filter can remove many candidates, and truncating first would leave too few.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from config import settings
from services.query_analyzer import QueryAnalysis, TemporalIntent
from services.retriever import CandidateChunk, assign_relevance_scores

log = logging.getLogger(__name__)

# ── Composite score weights for "current" questions (must sum to 1.0) ────────
W_RELEVANCE = 0.5
W_TEMPORAL  = 0.3
W_VERSION   = 0.1
W_LATEST    = 0.1

# Version boost applied when the user expressed no version preference.
NEUTRAL_VERSION_BOOST = 0.3

# Intents whose filter is "valid today".
_PRESENT_INTENTS = frozenset({TemporalIntent.CURRENT, TemporalIntent.ATEMPORAL})


def half_life_days() -> int:
    """
    The temporal decay half-life, in days.

    Read from settings on every call so tests (and a future admin setting) can
    override it without reimporting the module. `config.settings` is the single
    source of truth — see config.py.
    """
    return settings.temporal_half_life_days


def uses_recency(query_analysis: QueryAnalysis) -> bool:
    """True when freshness contributes to the score (current questions only)."""
    return query_analysis.intent is TemporalIntent.CURRENT


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
        now:            Reference time for "today" and for ages. Defaults to the
                        current UTC time; injectable so tests are deterministic.

    Returns:
        The surviving candidates, sorted best-first, with ``relevance_score``
        re-normalised and ``composite_score`` populated. ``temporal_score``,
        ``version_boost`` and ``latest_bonus`` are populated only when recency
        scoring applies, and are None otherwise. Not truncated — the caller
        decides how many to keep.
    """
    now = _as_utc(now or datetime.now(timezone.utc))
    today = now.date()

    kept = filter_by_validity(candidates, query_analysis, today)
    assign_relevance_scores(kept)

    recency = uses_recency(query_analysis)
    hl = half_life_days()
    for cand in kept:
        if recency:
            temporal_weight = temporal_weight_for(cand.valid_from, now, hl)
            version_boost = version_boost_for(cand.version_string, query_analysis.version_hint)
            latest_bonus = 1.0 if cand.is_latest else 0.0
            composite = (
                W_RELEVANCE * cand.relevance_score
                + W_TEMPORAL * temporal_weight
                + W_VERSION  * version_boost
                + W_LATEST   * latest_bonus
            )
            cand.temporal_score = round(temporal_weight, 4)
            cand.version_boost = round(version_boost, 4)
            cand.latest_bonus = round(latest_bonus, 4)
        else:
            composite = cand.relevance_score
            cand.temporal_score = cand.version_boost = cand.latest_bonus = None
        cand.composite_score = round(composite, 4)

    # chunk_id is a secondary key so ties produce a stable, reproducible order
    # across runs — a coin-flip ordering would make evaluation irreproducible.
    kept.sort(key=lambda c: (-(c.composite_score or 0.0), c.chunk_id))

    log.info(
        "rerank: %d → %d candidates (intent=%s, filter=%s, scoring=%s)",
        len(candidates), len(kept), query_analysis.intent.value,
        describe_filter(query_analysis, today), scoring_label(query_analysis),
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

    if intent in _PRESENT_INTENTS:
        # is_superseded and valid_to are set together by version_resolver;
        # checking the flag as well keeps superseded text out of current
        # answers even for a row whose valid_to was never stamped.
        return not chunk.is_superseded and valid_on(chunk, today)

    if intent is TemporalIntent.POINT_IN_TIME:
        return valid_on(chunk, query_analysis.as_of)

    if intent is TemporalIntent.RANGE:
        return overlaps(chunk, query_analysis.start_date, query_analysis.end_date)

    if intent is TemporalIntent.VERSION:
        return version_boost_for(chunk.version_string, query_analysis.version_hint) == 1.0

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
    if a.intent in _PRESENT_INTENTS:
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
        return f"from version {a.version_hint}"
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
        return f"version {a.version_hint} of the documents"
    if a.intent is TemporalIntent.HISTORICAL:
        return "the past (sources from every version are included)"
    return None


def scoring_label(query_analysis: QueryAnalysis) -> str:
    return "relevance + recency" if uses_recency(query_analysis) else "relevance only"


# ── Scoring helpers ──────────────────────────────────────────────────────────


def temporal_weight_for(
    valid_from: datetime | None,
    now: datetime,
    half_life: int | None = None,
) -> float:
    """
    Exponential decay: ``2 ** (-age_days / half_life)``.

    If ``valid_from`` is None (chunk has no date) the weight is 1.0 — undated
    content is treated as maximally fresh rather than being unfairly penalised.
    Future-dated content also scores 1.0 (age is clamped at 0).
    """
    if valid_from is None:
        return 1.0

    hl = half_life if half_life is not None else half_life_days()
    if hl <= 0:
        return 1.0

    age_days = max(0, (_as_utc(now) - _as_utc(valid_from)).days)
    return 2.0 ** (-age_days / hl)


def version_boost_for(chunk_version: str | None, version_hint: str | None) -> float:
    """
    Score how well a chunk's version matches the version the user asked about.

    Returns:
        NEUTRAL_VERSION_BOOST — the query named no version, so no preference
                                applies and every chunk is treated equally.
        1.0                   — the query named a version and this chunk matches.
        0.0                   — the query named a version and this chunk does
                                not match, or has no version at all.

    Matching rules:
        "1.x"  matches any chunk whose version starts with the "1." series.
        "2.0"  matches "2.0" and "2.0.1" (prefix match on components), but not
               "2.01" — comparison is component-wise, not string-prefix, so
               "2.1" does not spuriously match "2.10".
    """
    if version_hint is None:
        return NEUTRAL_VERSION_BOOST
    if chunk_version is None:
        return 0.0

    hint = version_hint.strip().lower().lstrip("v")
    chunk = chunk_version.strip().lower().lstrip("v")
    if not hint or not chunk:
        return 0.0

    hint_parts = hint.replace("_", ".").split(".")
    chunk_parts = chunk.replace("_", ".").split(".")

    # A wildcard component ("1.x") matches any value in that position and
    # ignores everything after it.
    for i, hint_part in enumerate(hint_parts):
        if hint_part == "x":
            return 1.0
        if i >= len(chunk_parts):
            # Hint is more specific than the chunk version: "2.0.1" vs "2.0".
            # Missing components are implicitly zero, matching the version
            # ordering policy in services/version_resolver.py.
            if hint_part != "0":
                return 0.0
            continue
        if not _component_eq(hint_part, chunk_parts[i]):
            return 0.0

    return 1.0


def _component_eq(a: str, b: str) -> bool:
    """Compare two version components numerically when possible, else literally."""
    if a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return a == b


def _as_utc(d: datetime) -> datetime:
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _day(d: datetime | None) -> date | None:
    """The UTC calendar day of a timestamp."""
    return None if d is None else _as_utc(d).astimezone(timezone.utc).date()
