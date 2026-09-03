"""
services/temporal_reranker.py — Temporal-aware candidate reranker.

rerank(candidates, query_analysis) → list[CandidateChunk]

Scoring formula
---------------
For each candidate chunk:

    age_days        = (today - chunk.valid_from).days   [0 if valid_from is None]
    temporal_weight = 2 ** (-age_days / half_life_days)

    version_boost:
        1.0  if query has a version_hint AND chunk version matches
        0.3  if query has NO version_hint  (neutral — no preference expressed)
        0.0  if query has a version_hint AND chunk version does NOT match

    latest_bonus    = 1.0 if doc.is_latest else 0.0

    composite = (
        0.5 * relevance_score      ← fused BM25+dense relevance, normalised [0,1]
      + 0.3 * temporal_weight
      + 0.1 * version_boost
      + 0.1 * latest_bonus
    )

Why `relevance_score` and not `semantic_score`
----------------------------------------------
This term previously used `semantic_score`, which had two problems:

  1. It discarded the BM25 signal entirely, throwing away the result of the
     hybrid retrieval + RRF fusion the pipeline had just computed.
  2. In its old `1/(1+d)` form it spanned only ~0.40-0.43 in practice, so
     `0.5 * semantic` varied by ~0.01 while `0.3 * temporal` varied by 0.3.
     Relevance was nominally weighted highest but contributed ~2% of the
     ranking variance — the reranker was, in effect, sorting by date alone.

`relevance_score` is the RRF score min-max normalised across the candidate set
(see services/retriever.py), so it carries both retrievers' signal and actually
spans [0, 1]. The documented weights now mean what they say.

`latest_bonus` is 1.0/0.0 (scaled by W_LATEST) rather than the previous
0.1/0.0, which was multiplied by W_LATEST again and so contributed at most
0.01 — an order of magnitude less than the comment claimed.

Pre-filter
----------
Chunks with is_superseded=True are removed UNLESS the query wants historical
sources — i.e. the user used historical wording OR pinned a specific version
(see QueryAnalysis.wants_historical_sources).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import settings
from services.query_analyzer import QueryAnalysis
from services.retriever import CandidateChunk

log = logging.getLogger(__name__)

# ── Composite score weights (must sum to 1.0) ────────────────────────────────
W_RELEVANCE = 0.5
W_TEMPORAL  = 0.3
W_VERSION   = 0.1
W_LATEST    = 0.1

# Version boost applied when the user expressed no version preference.
NEUTRAL_VERSION_BOOST = 0.3


def half_life_days() -> int:
    """
    The temporal decay half-life, in days.

    Read from settings on every call so tests (and a future admin setting) can
    override it without reimporting the module. `config.settings` is the single
    source of truth — see config.py.
    """
    return settings.temporal_half_life_days


def rerank(
    candidates: list[CandidateChunk],
    query_analysis: QueryAnalysis,
    now: datetime | None = None,
) -> list[CandidateChunk]:
    """
    Rerank *candidates* using fused relevance + temporal freshness + version awareness.

    Args:
        candidates:     Output of :func:`~services.retriever.hybrid_retrieve`.
        query_analysis: Output of :func:`~services.query_analyzer.analyze`.
        now:            Reference time for age computation. Defaults to the
                        current UTC time; injectable so tests are deterministic.

    Returns:
        Sorted list of the same :class:`CandidateChunk` objects with
        ``temporal_score``, ``version_boost``, ``latest_bonus`` and
        ``composite_score`` populated. Superseded chunks may be filtered out,
        so the returned list can be shorter than the input.
    """
    today = now or datetime.now(timezone.utc)
    version_hint = query_analysis.version_hint
    allow_superseded = query_analysis.wants_historical_sources
    hl = half_life_days()

    kept: list[CandidateChunk] = []

    for cand in candidates:
        # ── Pre-filter ───────────────────────────────────────────────────────
        if cand.is_superseded and not allow_superseded:
            log.debug(
                "rerank: dropping superseded chunk %s (doc=%s v=%s)",
                cand.chunk_id[:8], cand.doc_id[:8], cand.version_string,
            )
            continue

        temporal_weight = temporal_weight_for(cand.valid_from, today, hl)
        version_boost = version_boost_for(cand.version_string, version_hint)
        latest_bonus = 1.0 if cand.is_latest else 0.0

        composite = (
            W_RELEVANCE * cand.relevance_score
            + W_TEMPORAL * temporal_weight
            + W_VERSION  * version_boost
            + W_LATEST   * latest_bonus
        )

        cand.temporal_score  = round(temporal_weight, 4)
        cand.version_boost   = round(version_boost, 4)
        cand.latest_bonus    = round(latest_bonus, 4)
        cand.composite_score = round(composite, 4)

        kept.append(cand)

    # Sort by composite descending. chunk_id is a secondary key so that ties
    # produce a stable, reproducible order across runs — this matters for
    # evaluation, where a coin-flip ordering would make results irreproducible.
    kept.sort(key=lambda c: (-(c.composite_score or 0.0), c.chunk_id))

    log.info(
        "rerank: %d → %d candidates (version_hint=%r, allow_superseded=%s, half_life=%dd)",
        len(candidates), len(kept), version_hint, allow_superseded, hl,
    )
    return kept


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

    # Ensure timezone-aware comparison
    if valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_days = max(0, (now - valid_from).days)
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
