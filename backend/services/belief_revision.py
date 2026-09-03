"""
services/belief_revision.py — Belief Revision Engine.

revise(candidates, conflicts, db) -> RevisionResult

Decision tree (applied per conflict pair):
  1. Pre-resolved in DB        → apply the stored resolution_type
                                  (see _apply_stored_resolution for the
                                   action/confidence mapping per type)
  2. version_supersession      → exclude older chunk; high confidence
  3. direct_contradiction      → if gap > VERSION_SUPERSESSION_GAP_DAYS: prefer newer (medium)
                                  if gap <= VERSION_SUPERSESSION_GAP_DAYS: include both (low)
  4. unknown / future types    → include both; low confidence (safe fallback)

Every branch returns a consistent (exclude, confidence) pair: a branch may only
report high confidence when it actually resolved the conflict by excluding a
side, or when including both sides IS the correct resolution.

NOTE: "scope_change" is intentionally NOT produced or handled. The Phase 6
classifier only emits version_supersession or direct_contradiction, so a
scope_change branch here would be dead code for a type nothing generates.
(The manual resolution_type "scope_clarification" is a different thing — a
human asserting that two sources address different scopes — and IS handled.)

The VERSION_SUPERSESSION_GAP_DAYS threshold is imported from conflict_detector
so both phases always use the identical value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = ["RevisionResult", "revise"]

from sqlalchemy.orm import Session

from models import ConflictPair
from services.conflict_detector import ConflictResult, VERSION_SUPERSESSION_GAP_DAYS
from services.query_analyzer import QueryAnalysis
from services.retriever import CandidateChunk
from services.temporal_reranker import version_boost_for

log = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class RevisionResult:
    """Output of the belief revision engine."""
    include_chunks: list[str]           # chunk_ids to include in main context
    exclude_chunks: list[str]           # chunk_ids demoted to footnote / deprecated
    conflict_notices: list[str]         # injected notice strings for context.py
    answer_confidence: str              # "high" | "medium" | "low" | "none"
    confidence_reason: str              # plain-English explanation
    belief_revision_applied: bool       # True if any conflict was processed

    # chunk_ids that survived revision AND were party to a conflict. Only these
    # are genuinely "preferred over an alternative"; context.py marks exactly
    # this set, instead of labelling every retained chunk "← PREFERRED".
    preferred_chunks: list[str] = field(default_factory=list)

    # Per-chunk plain-English reason for exclusion, surfaced in the API so the
    # UI can explain why a retrieved source did not inform the answer.
    exclusion_reasons: dict[str, str] = field(default_factory=dict)


# ── Public entry-point ────────────────────────────────────────────────────────

def revise(
    candidates: list[CandidateChunk],
    conflicts: list[ConflictResult],
    db: Session,
    query_analysis: "QueryAnalysis | None" = None,
) -> RevisionResult:
    """
    Run belief revision over the retrieved candidates given detected conflicts.

    Args:
        candidates:     Full ranked list of CandidateChunk (post temporal-rerank).
        conflicts:      Conflict pairs from conflict_detector.detect().
        db:             Active SQLAlchemy session (loads stored resolutions).
        query_analysis: What the user asked for. When the query pins a specific
                        version, chunks from that version are protected from
                        exclusion — see _protected_chunk_ids.

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

    # Chunks the user explicitly asked for, which must never be excluded.
    protected = _protected_chunk_ids(candidates, query_analysis)

    exclude_set: set[str] = set()
    exclusion_reasons: dict[str, str] = {}
    conflicted_ids: set[str] = set()
    notices: list[str] = []
    confidence_levels: list[str] = []   # collect per-pair, take worst at end

    for conflict in conflicts:
        id_a = conflict.chunk_id_a
        id_b = conflict.chunk_id_b
        conflicted_ids.update((id_a, id_b))

        # ── 1. Check for stored resolution ─────────────────────────────────
        stored = _load_stored_resolution(db, id_a, id_b)
        if stored is not None:
            result = _apply_stored_resolution(stored, chunk_map, id_a, id_b)
        else:
            result = _apply_decision_tree(conflict, chunk_map, id_a, id_b)

        for cid in result["exclude"]:
            if cid in protected:
                # The user pinned this version. Dropping it would answer a
                # different question than the one that was asked.
                log.info(
                    "belief_revision: keeping chunk %s despite conflict — the "
                    "query pinned version %r.",
                    cid[:8], query_analysis.version_hint if query_analysis else None,
                )
                notices.append(
                    "NOTE: A newer source contradicts the version the question "
                    "asked about. Both are included because the question named "
                    "a specific version — report what that version says, and "
                    "note that it has since changed."
                )
                continue
            exclude_set.add(cid)
            exclusion_reasons.setdefault(cid, result["exclusion_reason"])
        notices.extend(result["notices"])
        confidence_levels.append(result["confidence"])

    # ── Aggregate confidence: take the worst level seen ───────────────────
    final_confidence = _worst_confidence(confidence_levels)
    final_reason = _build_reason(confidence_levels, conflicts)

    include_ids = [cid for cid in all_chunk_ids if cid not in exclude_set]

    # Only chunks that both survived revision AND were party to a conflict are
    # meaningfully "preferred" — the rest were simply never contested.
    preferred = [cid for cid in include_ids if cid in conflicted_ids]

    log.info(
        "belief_revision: %d conflicts processed → confidence=%s, "
        "%d chunks excluded, %d preferred",
        len(conflicts), final_confidence, len(exclude_set), len(preferred),
    )

    return RevisionResult(
        include_chunks=include_ids,
        exclude_chunks=list(exclude_set),
        conflict_notices=notices,
        answer_confidence=final_confidence,
        confidence_reason=final_reason,
        belief_revision_applied=True,
        preferred_chunks=preferred,
        exclusion_reasons=exclusion_reasons,
    )


# ── Query-intent protection ───────────────────────────────────────────────────

def _protected_chunk_ids(
    candidates: list[CandidateChunk],
    query_analysis: QueryAnalysis | None,
) -> set[str]:
    """
    Chunks that belief revision must not exclude, because the user asked for
    exactly that version.

    Without this, the two halves of the temporal pipeline fight each other:
    the reranker correctly surfaces the v1.0 passage for "how did buffering
    work in v1.0?", and belief revision then drops it as superseded, so the
    answer is built from v2.0 and describes the version the user did NOT ask
    about. The user's explicit version wins — the contradiction is still
    reported in the notices, it just isn't resolved by deletion.

    Only an explicit version pin protects a chunk. Vague historical wording
    ("what was the old way?") does not, because there is no specific version to
    honour and normal supersession handling is the right behaviour there.
    """
    if query_analysis is None or query_analysis.version_hint is None:
        return set()

    hint = query_analysis.version_hint
    return {
        c.chunk_id
        for c in candidates
        if version_boost_for(c.version_string, hint) == 1.0
    }


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
            "exclusion_reason": "",
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
        f"by {gap_days} days. The newer source is PREFERRED and should be used "
        f"for the current answer."
        + _superseded_footnote(older_chunk, older_title, older_date)
    )

    return {
        "exclude": [older_id],
        "exclusion_reason": (
            f"Superseded by {newer_title} ({newer_date}), {gap_days} days newer."
        ),
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
            f"and is PREFERRED."
            + _superseded_footnote(older_chunk, older_title, older_date)
        )
        return {
            "exclude": [older_id],
            "exclusion_reason": (
                f"Contradicts {newer_title} ({newer_date}), which is "
                f"{gap_days} days newer."
            ),
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
            "exclusion_reason": "",
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
    Apply a previously stored resolution, skipping decision-tree re-reasoning.

    Each resolution type maps to a CONSISTENT (action, confidence) pair. The
    previous implementation always returned an empty exclude list while
    reporting "high" confidence for every type, which was self-contradictory in
    two ways: resolving a conflict made retrieval strictly worse than leaving it
    unresolved (the decision tree would have excluded the stale chunk, the
    stored path did not), and it told the LLM to be highly confident while
    handing it both sides of an unreconciled contradiction.

    Mapping:
      temporal_preference  → the resolver chose "newer wins", so exclude the
                             older chunk. Confidence high — the action is known.
      scope_clarification  → the claims do not actually conflict; they apply to
                             different scopes. Keep both, labelled. Confidence
                             high — including both is the correct outcome.
      manual / anything else → a human adjudicated but did not record which side
                             won, so no automatic exclusion is possible. Keep
                             both and surface the note. Confidence medium — we
                             cannot act on the decision, only report it.
    """
    res_type = row.resolution_type or "manual"
    note = row.resolution_note or ""

    older_id, _newer_id, older_chunk, newer_chunk = _order_by_date(
        chunk_map.get(id_a), chunk_map.get(id_b), id_a, id_b
    )

    exclusion_reason = ""
    if res_type == "temporal_preference":
        exclude = [older_id]
        confidence = "high"
        detail = (
            f"{_title(newer_chunk)} ({_fmt_date(newer_chunk)}) was chosen over "
            f"{_title(older_chunk)} ({_fmt_date(older_chunk)}); the older claim "
            f"is excluded."
        )
        exclusion_reason = (
            f"A reviewer resolved this conflict in favour of "
            f"{_title(newer_chunk)} ({_fmt_date(newer_chunk)})."
        )
    elif res_type == "scope_clarification":
        exclude = []
        confidence = "high"
        detail = (
            "The two sources were judged to apply to different scopes rather "
            "than to contradict each other, so both are included."
        )
    else:
        exclude = []
        confidence = "medium"
        detail = (
            "A reviewer resolved this conflict manually without recording which "
            "source supersedes the other, so both are included."
        )

    notice = (
        f"✓ CONFLICT PREVIOUSLY RESOLVED (type: {res_type}).\n{detail}"
        + (f"\nReviewer note: {note}" if note else "")
    )

    log.debug(
        "belief_revision: using stored resolution for (%s, %s): type=%s, excluding %d",
        id_a[:8], id_b[:8], res_type, len(exclude),
    )

    return {
        "exclude": exclude,
        "exclusion_reason": exclusion_reason,
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

# How much of a superseded claim to quote back into the context.
FOOTNOTE_CHARS = 400


def _superseded_footnote(
    older_chunk: Optional[CandidateChunk],
    older_title: str,
    older_date: str,
) -> str:
    """
    Quote the superseded claim so the model can describe what changed.

    An excluded chunk is removed from the context entirely, so without this the
    model is told that an older source was overridden but never shown what it
    said. Asked "was this previously synchronous, and is it still?", the model
    could only answer the second half — it explicitly reported that the older
    text was not among its sources.

    The quote is deliberately short and labelled as superseded: enough to state
    what the old behaviour was, not enough to compete with the preferred source.
    """
    if older_chunk is None or not older_chunk.content:
        return ""

    quote = " ".join(older_chunk.content.split())
    if len(quote) > FOOTNOTE_CHARS:
        quote = quote[:FOOTNOTE_CHARS].rstrip() + "…"

    return (
        f"\nThe superseded {older_title} ({older_date}) stated: \"{quote}\"\n"
        f"Use this only to describe what CHANGED. Do not present it as current."
    )

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
