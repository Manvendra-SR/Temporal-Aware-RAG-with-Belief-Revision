"""
services/conflict_detector.py — NLI-based conflict detection pipeline.

detect(candidates, db, query_id) → list[ConflictResult]

Pipeline
--------
Step 1  Pair selection      — shared entity OR cosine sim > 0.55  (avoids O(n²) NLI)
Step 2  Cache check         — query conflict_pairs for existing canonical pairs
Step 3  Batch NLI           — predict contradiction score for uncached pairs
Step 4  Classify            — direct_contradiction | version_supersession
Step 5  Persist             — INSERT … ON CONFLICT DO NOTHING (fully idempotent)

Idempotency guarantee
---------------------
Every pair is stored in canonical order  (min(id), max(id))  so (A,B) and (B,A)
map to the same row.  The database has a UniqueConstraint("chunk_id_a","chunk_id_b"),
and we use ON CONFLICT DO NOTHING to prevent duplicates even under concurrent
requests.  The cache check also uses canonical IDs so existing pairs are always found.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from itertools import combinations

import numpy as np
from sqlalchemy import insert, tuple_ as sa_tuple
from sqlalchemy.orm import Session

from models import ConflictPair
from services import nli
from services.retriever import CandidateChunk

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

NLI_THRESHOLD: float = 0.70        # P(contradiction) → direct_contradiction
COSINE_SIM_THRESHOLD: float = 0.55  # Pair selection: dense similarity gate
VERSION_SUPERSESSION_GAP_DAYS: int = 90  # Days gap for version_supersession

# Regex: named entities — CamelCase words, dotted names, Version X.Y, ALL_CAPS consts
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|v?\d+\.\d+(?:\.\d+)?|[A-Z]{2,})\b"
)


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class ConflictResult:
    chunk_id_a: str
    chunk_id_b: str
    conflict_type: str          # "direct_contradiction" | "version_supersession"
    nli_score: float
    is_cached: bool = False     # True = loaded from DB, not re-computed


# ── Public entry-point ────────────────────────────────────────────────────────


def detect(
    candidates: list[CandidateChunk],
    db: Session,
    query_id: str | None = None,
) -> list[ConflictResult]:
    """
    Run the full conflict detection pipeline on *candidates*.

    Args:
        candidates:  Reranked CandidateChunk list (top-20 recommended).
        db:          Active SQLAlchemy session.
        query_id:    The current query UUID (stored as detected_during).

    Returns:
        List of ConflictResult objects (cached + newly detected).
    """
    if len(candidates) < 2:
        return []

    t_start = time.monotonic()

    # ── Step 1: Pair selection ────────────────────────────────────────────────
    candidate_pairs: list[tuple[CandidateChunk, CandidateChunk]] = _select_pairs(candidates)

    if not candidate_pairs:
        log.debug("conflict_detector: no candidate pairs after selection.")
        return []

    canonical_keys = [_canonical(a.chunk_id, b.chunk_id) for a, b in candidate_pairs]

    # ── Step 2: Cache check ───────────────────────────────────────────────────
    cached_map = _load_cached(db, canonical_keys)

    uncached_pairs = [
        (a, b)
        for (a, b), key in zip(candidate_pairs, canonical_keys)
        if key not in cached_map
    ]

    # ── Step 3: Batch NLI ────────────────────────────────────────────────────
    new_results: list[ConflictResult] = []
    if uncached_pairs and nli.is_loaded():
        t_nli = time.monotonic()
        text_pairs = [(a.content, b.content) for a, b in uncached_pairs]
        scores = nli.predict(text_pairs)
        nli_elapsed = (time.monotonic() - t_nli) * 1000
        log.info(
            "conflict_detector: NLI step %.0f ms, pairs evaluated: %d",
            nli_elapsed, len(uncached_pairs),
        )

        # ── Step 4: Classify ──────────────────────────────────────────────────
        today = datetime.now(timezone.utc)
        for (a, b), score in zip(uncached_pairs, scores):
            conflict_type = _classify(a, b, score, today)
            if conflict_type is None:
                continue
            id_a, id_b = _canonical(a.chunk_id, b.chunk_id)
            new_results.append(ConflictResult(
                chunk_id_a=id_a,
                chunk_id_b=id_b,
                conflict_type=conflict_type,
                nli_score=round(float(score), 4),
                is_cached=False,
            ))
    elif not nli.is_loaded():
        log.warning("conflict_detector: NLI model not loaded, skipping NLI step.")

    # ── Step 5: Persist (idempotent) ──────────────────────────────────────────
    if new_results:
        _persist(db, new_results, query_id)

    # Combine cached + new
    all_results: list[ConflictResult] = list(cached_map.values()) + new_results

    elapsed = (time.monotonic() - t_start) * 1000
    log.info(
        "conflict_detector: total %.0f ms — %d cached, %d new, %d conflicts",
        elapsed, len(cached_map), len(new_results), len(all_results),
    )
    return all_results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _canonical(id_a: str, id_b: str) -> tuple[str, str]:
    """
    Return the canonical ordered pair (smaller_uuid, larger_uuid).

    This ensures (A,B) and (B,A) always map to the same key and DB row.
    """
    return (min(id_a, id_b), max(id_a, id_b))


def _select_pairs(
    candidates: list[CandidateChunk],
) -> list[tuple[CandidateChunk, CandidateChunk]]:
    """
    Reduce O(n²) pairs to a manageable subset.

    A pair (A, B) is selected if:
      • They share at least one named entity (CamelCase / version / ALL_CAPS), OR
      • Their content vectors have cosine similarity > COSINE_SIM_THRESHOLD.

    This typically cuts 190 pairs (n=20) down to 10–30.
    """
    selected = []
    for a, b in combinations(candidates, 2):
        if _shares_entity(a.content, b.content):
            selected.append((a, b))
            continue
        sim = _cosine_sim(a.content, b.content)
        if sim > COSINE_SIM_THRESHOLD:
            selected.append((a, b))
    return selected


def _shares_entity(text_a: str, text_b: str) -> bool:
    """Return True if both texts share at least one named entity token."""
    entities_a = set(_ENTITY_RE.findall(text_a))
    entities_b = set(_ENTITY_RE.findall(text_b))
    return bool(entities_a & entities_b)


def _cosine_sim(text_a: str, text_b: str) -> float:
    """
    Lightweight character-level bag-of-words cosine similarity.

    Not as accurate as sentence embeddings, but fast and dependency-free.
    Used only for pair *selection* (not for scoring).
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a) ** 0.5 * len(words_b) ** 0.5)


def _load_cached(
    db: Session,
    canonical_keys: list[tuple[str, str]],
) -> dict[tuple[str, str], ConflictResult]:
    """
    Query conflict_pairs for any already-stored (chunk_id_a, chunk_id_b) pairs.

    Returns a dict keyed by canonical tuple → ConflictResult(is_cached=True).
    """
    if not canonical_keys:
        return {}

    rows = (
        db.query(ConflictPair)
        .filter(
            sa_tuple(ConflictPair.chunk_id_a, ConflictPair.chunk_id_b).in_(canonical_keys)
        )
        .all()
    )

    result: dict[tuple[str, str], ConflictResult] = {}
    for row in rows:
        key = (row.chunk_id_a, row.chunk_id_b)
        result[key] = ConflictResult(
            chunk_id_a=row.chunk_id_a,
            chunk_id_b=row.chunk_id_b,
            conflict_type=row.conflict_type or "unknown",
            nli_score=row.nli_score or 0.0,
            is_cached=True,
        )
    return result


def _classify(
    a: CandidateChunk,
    b: CandidateChunk,
    nli_score: float,
    today: datetime,
) -> str | None:
    """
    Classify a pair given its NLI contradiction score.

    Returns a conflict_type string or None if below threshold.
    """
    if nli_score < NLI_THRESHOLD:
        return None

    # Prefer version_supersession when there is a clear time gap
    date_a = a.valid_from
    date_b = b.valid_from
    if date_a and date_b:
        if date_a.tzinfo is None:
            date_a = date_a.replace(tzinfo=timezone.utc)
        if date_b.tzinfo is None:
            date_b = date_b.replace(tzinfo=timezone.utc)
        gap_days = abs((date_a - date_b).days)
        if gap_days >= VERSION_SUPERSESSION_GAP_DAYS:
            return "version_supersession"

    return "direct_contradiction"


def _persist(
    db: Session,
    results: list[ConflictResult],
    query_id: str | None,
) -> None:
    """
    Insert ConflictPair rows, skipping any that already exist.

    Uses INSERT … ON CONFLICT DO NOTHING so this is safe to call
    concurrently from multiple requests without creating duplicates.
    """
    if not results:
        return

    now = datetime.now(timezone.utc)
    rows_to_insert = [
        {
            "conflict_id": _new_uuid(),
            "chunk_id_a": r.chunk_id_a,
            "chunk_id_b": r.chunk_id_b,
            "conflict_type": r.conflict_type,
            "nli_score": r.nli_score,
            "detected_at": now,
            "detected_during": query_id,
            "is_resolved": False,
        }
        for r in results
    ]

    try:
        stmt = (
            insert(ConflictPair)
            .values(rows_to_insert)
            .on_conflict_do_nothing(index_elements=["chunk_id_a", "chunk_id_b"])
        )
        db.execute(stmt)
        db.commit()
        log.debug("conflict_detector: persisted %d new conflict rows.", len(rows_to_insert))
    except Exception as exc:
        log.warning("conflict_detector: failed to persist conflicts: %s", exc)
        db.rollback()


def _new_uuid() -> str:
    import uuid
    return str(uuid.uuid4())
