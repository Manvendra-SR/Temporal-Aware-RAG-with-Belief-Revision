"""
services/belief_revision.py — Conflict arbitration ("belief revision").

revise(candidates, conflicts, db, query_analysis) -> RevisionResult

The conflict detector says WHICH retrieved chunks contradict and whether the
two come from the same version lineage. This module decides what the answer
does about each contradiction: which chunks reach the LLM, how each conflict is
explained to it, and how confident the answer may claim to be.

Invariant
---------
Evidence is only ever EXCLUDED for questions about the present (intent
"current"). For every other intent the temporal filter has already
restricted the candidates to the time window the question is about,
so a disagreement inside that window is something to explain, not something
to delete — "who was CEO in 2023?" may legitimately need both the chunk that
names Rahul and the one that names Priya.

Rules, per conflict pair
------------------------
1. version_supersession — both chunks are versions of the same document.
     Keep both, labelled with their validity windows (high).
     The disagreement is the document changing over time, and lineage fully
     explains it, so it is never reviewed and no stored resolution applies.
     For present questions the pair cannot occur: a lineage's validity
     windows do not overlap, so only one version is valid today.

2. direct_contradiction with a reviewer's decision (Conflicts page). The
   decision replaces the automatic rule below:
     prefer_source (preferred_chunk_id names the winner)
       present: exclude the other side, quoting it as set aside   (high)
       past:    keep both, telling the model whom the reviewer trusts where
                they overlap — the other side may still be right about its
                own period, and deleting it would lose history          (high)
     both_valid  — not really a contradiction (different scopes, or an NLI
                   false positive); keep both                          (high)
   A row whose decision is malformed is ignored, so rule 3 applies.

3. direct_contradiction, pending review — two unrelated documents disagree.
     present, both dated, published more than CONTRADICTION_RECENCY_GAP_DAYS
       apart → prefer the newer, exclude the older (medium)
     otherwise (close or identical dates, a missing date, or a question about
       the past) → keep both, flag the disagreement (low)

The final confidence is the worst across all pairs; "none" means no conflict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

__all__ = ["CONTRADICTION_RECENCY_GAP_DAYS", "RevisionResult", "revise"]

from sqlalchemy.orm import Session

from models import BOTH_VALID, VERSION_CHANGE, ConflictPair
from services.conflict_detector import ConflictResult
from services.query_analyzer import QueryAnalysis, TemporalIntent
from services.retriever import CandidateChunk
from services.version_resolver import version_sort_key

log = logging.getLogger(__name__)

# When two UNRELATED documents that are both valid today contradict each other,
# the newer one is preferred only if it was published more than this many days
# after the older one. Closer than that, publication order is too weak a signal
# to discard either side. A fixed, documented heuristic — not a calibrated value.
CONTRADICTION_RECENCY_GAP_DAYS = 90


@dataclass
class RevisionResult:
    """Output of the belief revision engine."""
    include_chunks: list[str]           # chunk_ids to include in main context
    exclude_chunks: list[str]           # chunk_ids removed from the context
    conflict_notices: list[str]         # injected notice strings for context.py
    answer_confidence: str              # "high" | "medium" | "low" | "none"
    confidence_reason: str              # plain-English explanation
    belief_revision_applied: bool       # True if any conflict was processed

    # chunk_ids that won a conflict — the other side was excluded in their
    # favour. context.py marks exactly this set "← PREFERRED". Conflicts that
    # keep both sides prefer neither, so the label never implies a ruling that
    # was not made.
    preferred_chunks: list[str] = field(default_factory=list)

    # Per-chunk plain-English reason for exclusion, surfaced in the API so the
    # UI can explain why a retrieved source did not inform the answer.
    exclusion_reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class _Ruling:
    """The outcome of arbitrating one conflict pair."""
    confidence: str
    notice: str
    exclude: Optional[str] = None       # chunk_id to drop, if any
    prefer: Optional[str] = None        # chunk_id that won, if any
    exclusion_reason: str = ""


# ── Public entry-point ────────────────────────────────────────────────────────

def revise(
    candidates: list[CandidateChunk],
    conflicts: list[ConflictResult],
    db: Session,
    query_analysis: QueryAnalysis | None = None,
) -> RevisionResult:
    """
    Arbitrate the detected conflicts among the retrieved candidates.

    Args:
        candidates:     Ranked candidates (after temporal filtering/reranking).
        conflicts:      Conflict pairs from conflict_detector.detect().
        db:             Active SQLAlchemy session (loads stored resolutions).
        query_analysis: How the question was interpreted. Omitted → treated as
                        a question about the present.

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

    chunk_map: dict[str, CandidateChunk] = {c.chunk_id: c for c in candidates}
    present = _is_about_present(query_analysis)

    rulings: list[_Ruling] = []
    for conflict in conflicts:
        a = chunk_map.get(conflict.chunk_id_a)
        b = chunk_map.get(conflict.chunk_id_b)
        if a is None or b is None:
            # A conflict about a chunk that is not among the candidates has
            # nothing to act on.
            continue

        if conflict.conflict_type == VERSION_CHANGE:
            rulings.append(_rule_version_change(a, b))
            continue
        # CONTRADICTION — the only other type the detector emits.
        stored = _load_stored_resolution(db, conflict.chunk_id_a, conflict.chunk_id_b)
        if stored is not None:
            rulings.append(_apply_stored_resolution(stored, a, b, present))
        else:
            rulings.append(_rule_unrelated_contradiction(a, b, present))

    if not rulings:
        return RevisionResult(
            include_chunks=all_chunk_ids,
            exclude_chunks=[],
            conflict_notices=[],
            answer_confidence="none",
            confidence_reason="No conflicts among the sources used.",
            belief_revision_applied=False,
        )

    exclude_set: set[str] = set()
    exclusion_reasons: dict[str, str] = {}
    for r in rulings:
        if r.exclude:
            exclude_set.add(r.exclude)
            exclusion_reasons.setdefault(r.exclude, r.exclusion_reason)

    include_ids = [cid for cid in all_chunk_ids if cid not in exclude_set]
    preferred = [
        cid for cid in include_ids
        if any(r.prefer == cid for r in rulings)
    ]
    levels = [r.confidence for r in rulings]
    final_confidence = _worst_confidence(levels)

    log.info(
        "belief_revision: %d conflicts (present=%s) → confidence=%s, "
        "%d chunks excluded, %d preferred",
        len(rulings), present, final_confidence, len(exclude_set), len(preferred),
    )

    return RevisionResult(
        include_chunks=include_ids,
        exclude_chunks=sorted(exclude_set),
        conflict_notices=[r.notice for r in rulings],
        answer_confidence=final_confidence,
        confidence_reason=_build_reason(final_confidence, len(rulings)),
        belief_revision_applied=True,
        preferred_chunks=preferred,
        exclusion_reasons=exclusion_reasons,
    )


def _is_about_present(query_analysis: QueryAnalysis | None) -> bool:
    return query_analysis is None or query_analysis.intent is TemporalIntent.CURRENT


# ── Rules ─────────────────────────────────────────────────────────────────────

def _rule_version_change(a: CandidateChunk, b: CandidateChunk) -> _Ruling:
    """Same lineage: the document changed between versions. Keep both."""
    older, newer = _order_by_version(a, b)
    notice = (
        f"ℹ VERSION CHANGE: {_title(older)} (valid {_window(older)}) and "
        f"{_title(newer)} (valid {_window(newer)}) are versions of the same "
        f"document and say different things. Both are included because each is "
        f"correct for its own validity period. Answer from the version(s) valid "
        f"for the time the question asks about, and say when it changed."
    )
    return _Ruling(confidence="high", notice=notice)


def _rule_unrelated_contradiction(
    a: CandidateChunk,
    b: CandidateChunk,
    present: bool,
) -> _Ruling:
    """Different lineages disagree. Only recency can break the tie, and only now."""
    older, newer = _order_by_date(a, b)
    gap = _gap_days(older, newer)

    if present and gap is not None and gap > CONTRADICTION_RECENCY_GAP_DAYS:
        notice = (
            f"⚠ CONFLICT (unrelated sources): {_title(older)} ({_fmt_date(older)}) "
            f"and {_title(newer)} ({_fmt_date(newer)}) make conflicting claims. "
            f"{_title(newer)} is {gap} days newer and is PREFERRED for the "
            f"current answer."
            + _excluded_footnote(older, by_reviewer=False)
        )
        return _Ruling(
            confidence="medium",
            notice=notice,
            exclude=older.chunk_id,
            prefer=newer.chunk_id,
            exclusion_reason=(
                f"Contradicts {_title(newer)} ({_fmt_date(newer)}), an unrelated "
                f"document published {gap} days later."
            ),
        )

    if not present:
        why = "the question is about the past, when either may have been accurate"
    elif gap is None:
        why = "at least one of them is undated"
    else:
        why = f"they were published only {gap} days apart"
    notice = (
        f"⚠ UNRESOLVED CONFLICT: {_title(a)} ({_fmt_date(a)}) and {_title(b)} "
        f"({_fmt_date(b)}) are unrelated documents that make conflicting claims, "
        f"and neither can be preferred because {why}. Both are included. "
        f"Present both claims and recommend verifying against primary sources."
    )
    return _Ruling(confidence="low", notice=notice)


# ── Stored resolution ─────────────────────────────────────────────────────────

def _load_stored_resolution(
    db: Session, id_a: str, id_b: str
) -> Optional[ConflictPair]:
    """
    Return the pair's row if a reviewer resolved it with a decision that can be
    applied; None otherwise (the automatic rule then applies).
    """
    row = (
        db.query(ConflictPair)
        .filter(
            ConflictPair.chunk_id_a == id_a,
            ConflictPair.chunk_id_b == id_b,
            ConflictPair.is_resolved == True,  # noqa: E712
        )
        .first()
    )
    if row is None:
        return None
    if not row.has_valid_resolution():
        log.warning(
            "belief_revision: ignoring malformed resolution on conflict %s "
            "(type=%r, preferred=%r)",
            row.conflict_id, row.resolution_type, row.preferred_chunk_id,
        )
        return None
    return row


def _apply_stored_resolution(
    row: ConflictPair,
    a: CandidateChunk,
    b: CandidateChunk,
    present: bool,
) -> _Ruling:
    """Apply a reviewer's decision — see rule 2 in the module docstring."""
    if row.resolution_type == BOTH_VALID:
        ruling = _Ruling(confidence="high", notice="")
        detail = (
            f"A reviewer judged that {_title(a)} ({_fmt_date(a)}) and {_title(b)} "
            f"({_fmt_date(b)}) do not actually contradict each other: they apply to "
            f"different scopes or situations. Both are included; use each for what "
            f"it covers."
        )
    else:  # PREFER_SOURCE — has_valid_resolution() guarantees the id is a or b
        preferred, other = (a, b) if row.preferred_chunk_id == a.chunk_id else (b, a)
        ruling = _Ruling(confidence="high", notice="")
        if present:
            detail = (
                f"A reviewer judged {_title(preferred)} ({_fmt_date(preferred)}) "
                f"correct over {_title(other)} ({_fmt_date(other)}); the other "
                f"claim is excluded."
                + _excluded_footnote(other, by_reviewer=True)
            )
            ruling.exclude = other.chunk_id
            ruling.prefer = preferred.chunk_id
            ruling.exclusion_reason = (
                f"A reviewer resolved this conflict in favour of "
                f"{_title(preferred)} ({_fmt_date(preferred)})."
            )
        else:
            detail = (
                f"A reviewer judged {_title(preferred)} ({_fmt_date(preferred)}) "
                f"correct over {_title(other)} ({_fmt_date(other)}). The question "
                f"is about the past, so both are included: {_title(other)} (valid "
                f"{_window(other)}) may still describe its own period. Where the "
                f"two disagree about the same time, rely on {_title(preferred)}."
            )

    note = row.resolution_note or ""
    ruling.notice = (
        f"✓ REVIEWED CONFLICT:\n{detail}"
        + (f"\nReviewer note: {note}" if note else "")
    )
    return ruling


# ── Aggregation helpers ───────────────────────────────────────────────────────

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _worst_confidence(levels: list[str]) -> str:
    """Return the worst (lowest) confidence across all processed pairs."""
    if not levels:
        return "none"
    return min(levels, key=lambda l: _CONFIDENCE_RANK.get(l, 0))


def _build_reason(worst: str, n: int) -> str:
    if worst == "high":
        return (
            f"{n} conflict(s), each explained by the document's version history "
            f"or settled by a reviewer's decision."
        )
    if worst == "medium":
        return (
            f"{n} conflict(s). Where unrelated sources disagreed, the one published "
            f"more than {CONTRADICTION_RECENCY_GAP_DAYS} days later was preferred."
        )
    return (
        f"{n} conflict(s), at least one unresolved — conflicting sources are both "
        f"included. Verify against primary sources."
    )


# ── Formatting utilities ──────────────────────────────────────────────────────

# How much of an excluded claim to quote back into the context.
FOOTNOTE_CHARS = 400


def _excluded_footnote(excluded: CandidateChunk, *, by_reviewer: bool) -> str:
    """
    Quote the excluded claim so the model can describe the disagreement.

    An excluded chunk is removed from the context entirely, so without this the
    model is told that a source was overridden but never shown what it said,
    and cannot answer "what changed?".

    The quote is deliberately short and labelled: enough to state what the
    claim was, not enough to compete with the preferred source. A reviewer may
    prefer the OLDER source, so their exclusions are not called "superseded".
    """
    if not excluded.content:
        return ""

    quote = " ".join(excluded.content.split())
    if len(quote) > FOOTNOTE_CHARS:
        quote = quote[:FOOTNOTE_CHARS].rstrip() + "…"

    label = "set-aside" if by_reviewer else "superseded"
    use = (
        "A reviewer judged it incorrect; mention it only to explain the disagreement."
        if by_reviewer else
        "Use this only to describe what CHANGED."
    )
    return (
        f"\nThe {label} {_title(excluded)} ({_fmt_date(excluded)}) stated: \"{quote}\"\n"
        f"{use} Do not present it as current."
    )


def _order_by_version(
    a: CandidateChunk, b: CandidateChunk
) -> tuple[CandidateChunk, CandidateChunk]:
    """(older, newer) within one lineage, by the shared version ordering policy."""
    key_a, key_b = version_sort_key(a.version_string), version_sort_key(b.version_string)
    if key_a == key_b:
        return _order_by_date(a, b)
    return (a, b) if key_a < key_b else (b, a)


def _order_by_date(
    a: CandidateChunk, b: CandidateChunk
) -> tuple[CandidateChunk, CandidateChunk]:
    """(older, newer) by valid_from; undated or tied pairs keep their given order."""
    date_a, date_b = _get_date(a), _get_date(b)
    if date_a is not None and date_b is not None and date_b < date_a:
        return b, a
    return a, b


def _get_date(chunk: CandidateChunk) -> Optional[datetime]:
    d = chunk.valid_from
    if d is None:
        return None
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def _gap_days(older: CandidateChunk, newer: CandidateChunk) -> Optional[int]:
    """Days between the two publication dates, or None if either is undated."""
    d_old, d_new = _get_date(older), _get_date(newer)
    if d_old is None or d_new is None:
        return None
    return abs((d_new - d_old).days)


def _fmt_date(chunk: CandidateChunk) -> str:
    d = _get_date(chunk)
    return "unknown date" if d is None else d.strftime("%Y-%m-%d")


def _window(chunk: CandidateChunk) -> str:
    """Validity window as text: "2022-01-01 → 2023-03-15" or "since 2023-03-15"."""
    start = _fmt_date(chunk)
    if chunk.valid_to is None:
        return f"since {start}"
    return f"{start} → {chunk.valid_to.strftime('%Y-%m-%d')}"


def _title(chunk: CandidateChunk) -> str:
    parts = [chunk.doc_title]
    if chunk.version_string:
        parts.append(f"v{chunk.version_string}")
    return " ".join(parts)
