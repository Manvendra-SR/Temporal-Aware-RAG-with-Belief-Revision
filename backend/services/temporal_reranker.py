"""
services/temporal_reranker.py — Temporal-aware candidate reranker.

rerank(candidates, query_analysis, db) → list[CandidateChunk]

Scoring formula
---------------
For each candidate chunk:

    age_days        = (today - chunk.valid_from).days   [0 if valid_from is None]
    temporal_weight = 2 ** (-age_days / DEFAULT_HALF_LIFE_DAYS)

    version_boost:
        1.0  if query has a version_hint AND chunk version matches
        0.3  if query has NO version_hint  (neutral — no preference)
        0.0  if query has a version_hint AND chunk version does NOT match

    latest_bonus    = 0.1 if doc.is_latest else 0.0

    composite = (
        0.5 * semantic_score
      + 0.3 * temporal_weight
      + 0.1 * version_boost
      + 0.1 * latest_bonus
    )

Pre-filter
----------
Chunks with is_superseded=True are removed UNLESS the query has
temporal_qualifier=True (user explicitly asks about an older version).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from services.query_analyzer import QueryAnalysis

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_HALF_LIFE_DAYS: int = 180
"""Exponential half-life for temporal decay (in days)."""

# Composite score weights (must sum to 1.0)
W_SEMANTIC = 0.5
W_TEMPORAL = 0.3
W_VERSION  = 0.1
W_LATEST   = 0.1


# ── Main entry-point ─────────────────────────────────────────────────────────


def rerank(
    candidates: list,          # list[CandidateChunk] — avoid circular import
    query_analysis: QueryAnalysis,
    db: Session,               # noqa: ARG001  kept for future per-domain half-life lookup
) -> list:
    """
    Rerank *candidates* using temporal freshness + version awareness.

    Args:
        candidates:     Output of :func:`~services.retriever.hybrid_retrieve`.
        query_analysis: Output of :func:`~services.query_analyzer.analyze`.
        db:             Active SQLAlchemy session (reserved for future domain config).

    Returns:
        Sorted list of the same :class:`CandidateChunk` objects with
        ``temporal_score``, ``version_boost``, ``latest_bonus``, and
        ``composite_score`` populated.
    """
    today = datetime.now(timezone.utc)
    version_hint = query_analysis.version_hint  # e.g. "1.13" or None
    allow_superseded = query_analysis.temporal_qualifier

    scored: list[tuple[float, object]] = []

    for cand in candidates:
        # ── Pre-filter ───────────────────────────────────────────────────────
        if cand.is_superseded and not allow_superseded:
            log.debug(
                "rerank: skipping superseded chunk %s (doc=%s v=%s)",
                cand.chunk_id[:8], cand.doc_id[:8], cand.version_string,
            )
            continue

        # ── Temporal weight ──────────────────────────────────────────────────
        temporal_weight = _temporal_weight(cand.valid_from, today)

        # ── Version boost ────────────────────────────────────────────────────
        version_boost = _version_boost(cand.version_string, version_hint)

        # ── Latest bonus ─────────────────────────────────────────────────────
        latest_bonus = 0.1 if cand.is_latest else 0.0

        # ── Composite ────────────────────────────────────────────────────────
        composite = (
            W_SEMANTIC * cand.semantic_score
            + W_TEMPORAL * temporal_weight
            + W_VERSION  * version_boost
            + W_LATEST   * latest_bonus
        )

        # Attach scores directly to the dataclass
        cand.temporal_score   = round(temporal_weight, 4)
        cand.version_boost    = round(version_boost, 4)
        cand.latest_bonus     = round(latest_bonus, 4)
        cand.composite_score  = round(composite, 4)

        scored.append((composite, cand))

    # Sort descending by composite score
    scored.sort(key=lambda t: t[0], reverse=True)

    result = [cand for _, cand in scored]
    log.info(
        "rerank: %d → %d candidates after temporal rerank (version_hint=%r, allow_superseded=%s)",
        len(candidates), len(result), version_hint, allow_superseded,
    )
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────


def _temporal_weight(valid_from: datetime | None, today: datetime) -> float:
    """
    Exponential decay: ``2 ** (-age_days / DEFAULT_HALF_LIFE_DAYS)``.

    If ``valid_from`` is None (chunk has no date), age_days = 0 → weight = 1.0
    (treat undated content as maximally fresh so it isn't unfairly penalised).
    """
    if valid_from is None:
        return 1.0

    # Ensure timezone-aware comparison
    if valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=timezone.utc)

    age_days = max(0, (today - valid_from).days)
    return 2.0 ** (-age_days / DEFAULT_HALF_LIFE_DAYS)


def _version_boost(chunk_version: str | None, version_hint: str | None) -> float:
    """
    Compute version boost score.

    Returns:
        1.0  — query has a version hint AND chunk version matches
        0.3  — query has NO version hint (neutral)
        0.0  — query has a version hint AND chunk version does NOT match
    """
    if version_hint is None:
        return 0.3  # No preference expressed → neutral

    if chunk_version is None:
        return 0.0  # Hint given but chunk has no version → penalise slightly

    # Normalise both to lowercase, strip leading "v"
    hint_norm  = version_hint.lower().lstrip("v")
    chunk_norm = chunk_version.lower().lstrip("v")

    # Handle "x" wildcard: "1.x" matches "1.13", "1.0", etc.
    if hint_norm.endswith(".x"):
        prefix = hint_norm[:-2]  # e.g. "1"
        return 1.0 if chunk_norm.startswith(prefix) else 0.0

    # Exact prefix match: hint "2.0" matches chunk "2.0.1"
    return 1.0 if chunk_norm.startswith(hint_norm) else 0.0
