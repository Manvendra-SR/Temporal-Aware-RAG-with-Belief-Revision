"""
services/belief_revision.py — Belief Revision Engine.

revise(candidates, conflicts, db) -> RevisionResult

Decision tree (applied per conflict pair):
  1. Pre-resolved in DB        → use stored resolution_type, skip re-reasoning
  2. version_supersession      → exclude older chunk; high confidence
  3. direct_contradiction      → if gap > VERSION_SUPERSESSION_GAP_DAYS: prefer newer (medium)
                                  if gap <= VERSION_SUPERSESSION_GAP_DAYS: include both (low)
  4. unknown / future types   → include both; low confidence (safe fallback)

NOTE: scope_change is intentionally NOT handled here. Phase 6's classifier
only produces version_supersession or direct_contradiction. scope_change will
be introduced when scope-aware detection is added in a future phase.

The VERSION_SUPERSESSION_GAP_DAYS threshold is imported from conflict_detector
so both phases always use the identical value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from models import ConflictPair
from services.conflict_detector import ConflictResult, VERSION_SUPERSESSION_GAP_DAYS
from services.retriever import CandidateChunk

log = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class RevisionResult:
    """Output of the belief revision engine."""
    include_chunks: list[str]           # chunk_ids to include in main context
    exclude_chunks: list[str]           # chunk_ids demoted to footnote / deprecated
    conflict_notices: list[str]         # injected notice strings for context.py
    answer_confidence: str              # "high" | "medium" | "low" | "none"
    confidence_reason: str             # plain-English explanation
    belief_revision_applied: bool       # True if any conflict was processed


# ── Public entry-point ────────────────────────────────────────────────────────

def revise(
    candidates: list[CandidateChunk],
    conflicts: list[ConflictResult],
    db: Session,
) -> RevisionResult:
    """
    Run belief revision over the retrieved candidates given detected conflicts.

    Args:
        candidates:  Full ranked list of CandidateChunk (post temporal-rerank).
        conflicts:   Conflict pairs from conflict_detector.detect().
        db:          Active SQLAlchemy session (used to load stored resolutions).

    Returns:
        RevisionResult with include/exclude sets, notices, and confidence level.
    """
    all_chunk_ids = [c.chunk_id for c in candidates]

    if not conflicts:
        return RevisionResult(
            include_chunks=all_chunk_ids,
            exclude_chunks=[],
            conflict_notices=[],
            answer_confidence="none",
            confidence_reason="No conflicts detected.",
            belief_revision_applied=False,
        )

    # Map chunk_id → CandidateChunk for quick lookups
    chunk_map: dict[str, CandidateChunk] = {c.chunk_id: c for c in candidates}

    exclude_set: set[str] = set()
    notices: list[str] = []
    confidence_levels: list[str] = []   # collect per-pair, take worst at end

    for conflict in conflicts:
        id_a = conflict.chunk_id_a
        id_b = conflict.chunk_id_b

        # ── 1. Check for stored resolution ─────────────────────────────────
        stored = _load_stored_resolution(db, id_a, id_b)
        if stored is not None:
            result = _apply_stored_resolution(stored, chunk_map, id_a, id_b)
        else:
            result = _apply_decision_tree(conflict, chunk_map, id_a, id_b)

        exclude_set.update(result["exclude"])
        notices.extend(result["notices"])
        confidence_levels.append(result["confidence"])

    # ── Aggregate confidence: take the worst level seen ───────────────────
    final_confidence = _worst_confidence(confidence_levels)
    final_reason = _build_reason(confidence_levels, conflicts)

    include_ids = [cid for cid in all_chunk_ids if cid not in exclude_set]

    log.info(
        "belief_revision: %d conflicts processed → confidence=%s, %d chunks excluded",
        len(conflicts), final_confidence, len(exclude_set),
    )

    return RevisionResult(
        include_chunks=include_ids,
        exclude_chunks=list(exclude_set),
        conflict_notices=notices,
        answer_confidence=final_confidence,
        confidence_reason=final_reason,
        belief_revision_applied=True,
    )


# ── Decision tree helpers ─────────────────────────────────────────────────────

def _apply_decision_tree(
    conflict: ConflictResult,
    chunk_map: dict[str, CandidateChunk],
    id_a: str,
    id_b: str,
) -> dict:
    """Apply the conflict-type decision tree for a single pair."""
    chunk_a = chunk_map.get(id_a)
    chunk_b = chunk_map.get(id_b)

    if conflict.conflict_type == "version_supersession":
        return _handle_version_supersession(chunk_a, chunk_b, id_a, id_b)

    elif conflict.conflict_type == "direct_contradiction":
        return _handle_direct_contradiction(chunk_a, chunk_b, id_a, id_b)

    else:
        # Unknown / future conflict type — safe fallback: include both, low confidence
        log.debug(
            "belief_revision: unknown conflict_type=%r for pair (%s, %s), including both",
            conflict.conflict_type, id_a[:8], id_b[:8],
        )
        return {
            "exclude": [],
            "notices": [
                f"⚠ UNCLASSIFIED CONFLICT: Sources {_short(id_a)} and {_short(id_b)} "
                f"may contain inconsistent information (type: {conflict.conflict_type})."
            ],
            "confidence": "low",
        }


def _handle_version_supersession(
    chunk_a: Optional[CandidateChunk],
    chunk_b: Optional[CandidateChunk],
    id_a: str,
    id_b: str,
) -> dict:
    """
    version_supersession: exclude the older chunk, prefer the newer.
    Confidence: high.
    """
    older_id, newer_id, older_chunk, newer_chunk = _order_by_date(
        chunk_a, chunk_b, id_a, id_b
    )

    gap_days = _gap_days(older_chunk, newer_chunk)
    older_date = _fmt_date(older_chunk)
    newer_date = _fmt_date(newer_chunk)
    older_title = _title(older_chunk)
    newer_title = _title(newer_chunk)

    notice = (
        f"⚠ TEMPORAL CONFLICT DETECTED (version supersession):\n"
        f"{newer_title} ({newer_date}) supersedes {older_title} ({older_date}) "
        f"by {gap_days} days. "
        f"The newer source is PREFERRED. The older claim has been moved to footnote."
    )

    return {
        "exclude": [older_id],
        "notices": [notice],
        "confidence": "high",
    }


def _handle_direct_contradiction(
    chunk_a: Optional[CandidateChunk],
    chunk_b: Optional[CandidateChunk],
    id_a: str,
    id_b: str,
) -> dict:
    """
    direct_contradiction:
      - gap > VERSION_SUPERSESSION_GAP_DAYS → prefer newer (medium)
      - gap <= VERSION_SUPERSESSION_GAP_DAYS → include both (low)
    """
    older_id, newer_id, older_chunk, newer_chunk = _order_by_date(
        chunk_a, chunk_b, id_a, id_b
    )

    gap_days = _gap_days(older_chunk, newer_chunk)
    older_date = _fmt_date(older_chunk)
    newer_date = _fmt_date(newer_chunk)
    older_title = _title(older_chunk)
    newer_title = _title(newer_chunk)

    if gap_days > VERSION_SUPERSESSION_GAP_DAYS:
        # Clear temporal ordering — prefer newer
        notice = (
            f"⚠ TEMPORAL CONFLICT DETECTED (direct contradiction):\n"
            f"{older_title} ({older_date}) and {newer_title} ({newer_date}) "
            f"make conflicting claims. {newer_title} is more recent by {gap_days} days "
            f"and is PREFERRED. The older claim is marked [DEPRECATED]."
        )
        return {
            "exclude": [older_id],
            "notices": [notice],
            "confidence": "medium",
        }
    else:
        # Ambiguous timing — include both, flag both
        notice = (
            f"⚠ TEMPORAL CONFLICT DETECTED (ambiguous contradiction):\n"
            f"{older_title} ({older_date}) and {newer_title} ({newer_date}) "
            f"make conflicting claims with only a {gap_days}-day gap. "
            f"Both sources are included. Verify against primary sources."
        )
        return {
            "exclude": [],
            "notices": [notice],
            "confidence": "low",
        }


# ── Stored resolution ─────────────────────────────────────────────────────────

def _load_stored_resolution(
    db: Session, id_a: str, id_b: str
) -> Optional[ConflictPair]:
    """
    Return the ConflictPair row if it exists AND is_resolved=True.
    Returns None otherwise (triggers normal decision tree).
    """
    row = (
        db.query(ConflictPair)
        .filter(
            ConflictPair.chunk_id_a == id_a,
            ConflictPair.chunk_id_b == id_b,
            ConflictPair.is_resolved == True,
        )
        .first()
    )
    return row


def _apply_stored_resolution(
    row: ConflictPair,
    chunk_map: dict[str, CandidateChunk],
    id_a: str,
    id_b: str,
) -> dict:
    """
    Apply a previously stored resolution. Skip NLI/decision-tree re-reasoning.
    Maps resolution_type → confidence level.
    """
    res_type = row.resolution_type or "manual"
    note = row.resolution_note or ""

    confidence_map = {
        "temporal_preference": "high",
        "manual": "high",
        "scope_clarification": "high",
    }
    confidence = confidence_map.get(res_type, "medium")

    notice = (
        f"✓ CONFLICT PRE-RESOLVED (stored resolution):\n"
        f"Resolution type: {res_type}."
        + (f" Note: {note}" if note else "")
    )

    log.debug(
        "belief_revision: using stored resolution for (%s, %s): type=%s",
        id_a[:8], id_b[:8], res_type,
    )

    return {
        "exclude": [],     # stored resolutions don't exclude (trust the resolver's intent)
        "notices": [notice],
        "confidence": confidence,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}

def _worst_confidence(levels: list[str]) -> str:
    """Return the worst (lowest) confidence across all processed pairs."""
    if not levels:
        return "none"
    return min(levels, key=lambda l: _CONFIDENCE_RANK.get(l, 0))


def _build_reason(levels: list[str], conflicts: list[ConflictResult]) -> str:
    worst = _worst_confidence(levels)
    n = len(conflicts)

    if worst == "high":
        return f"All {n} conflict(s) resolved with high certainty by temporal ordering."
    elif worst == "medium":
        return (
            f"{n} conflict(s) detected. Newer sources preferred based on date ordering (>{VERSION_SUPERSESSION_GAP_DAYS} day gap)."
        )
    elif worst == "low":
        return (
            f"{n} conflict(s) detected with ambiguous or close dating. Both sources are included — verify against primary sources."
        )
    return "No conflicts processed."


# ── Formatting utilities ──────────────────────────────────────────────────────

def _order_by_date(
    chunk_a: Optional[CandidateChunk],
    chunk_b: Optional[CandidateChunk],
    id_a: str,
    id_b: str,
) -> tuple[str, str, Optional[CandidateChunk], Optional[CandidateChunk]]:
    """
    Return (older_id, newer_id, older_chunk, newer_chunk).
    Falls back to canonical ordering if dates are unavailable.
    """
    date_a = _get_date(chunk_a)
    date_b = _get_date(chunk_b)

    if date_a is not None and date_b is not None:
        if date_a <= date_b:
            return id_a, id_b, chunk_a, chunk_b
        else:
            return id_b, id_a, chunk_b, chunk_a

    # No dates — fall back to canonical (already min/max ordered)
    return id_a, id_b, chunk_a, chunk_b


def _get_date(chunk: Optional[CandidateChunk]) -> Optional[datetime]:
    if chunk is None:
        return None
    d = chunk.valid_from
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _gap_days(
    older: Optional[CandidateChunk],
    newer: Optional[CandidateChunk],
) -> int:
    d_old = _get_date(older)
    d_new = _get_date(newer)
    if d_old is None or d_new is None:
        return 0
    return abs((d_new - d_old).days)


def _fmt_date(chunk: Optional[CandidateChunk]) -> str:
    d = _get_date(chunk)
    if d is None:
        return "unknown date"
    return d.strftime("%Y-%m-%d")


def _title(chunk: Optional[CandidateChunk]) -> str:
    if chunk is None:
        return "Unknown source"
    parts = [chunk.doc_title]
    if chunk.version_string:
        parts.append(f"v{chunk.version_string}")
    return " ".join(parts)


def _short(chunk_id: str) -> str:
    return chunk_id[:8] + "…"
