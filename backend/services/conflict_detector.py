"""
services/conflict_detector.py — NLI-based conflict detection pipeline.

detect(candidates, db, query_id) → list[ConflictResult]

Pipeline
--------
Step 1  Pair selection      — shared entity OR lexical overlap > threshold
Step 2  Cache check         — query conflict_pairs for existing canonical pairs
Step 3  Batch NLI           — predict contradiction score for uncached pairs
Step 4  Classify            — version_supersession | direct_contradiction
Step 5  Persist             — INSERT … ON CONFLICT DO NOTHING (fully idempotent)

Classification
--------------
NLI decides WHETHER two chunks contradict. Lineage decides WHAT KIND of
contradiction it is:

    version_supersession   both chunks come from the same version lineage
                           (one document is a version of the other), so the
                           disagreement is the document changing over time
    direct_contradiction   the chunks come from unrelated documents, so nothing
                           establishes which one is authoritative

Dates play no part in the type. An earlier design called any contradiction
between chunks ≥ 90 days apart a "version_supersession", which labelled two
unrelated documents as versions of each other whenever they happened to be
published a few months apart — and belief revision then resolved them with
high confidence on the strength of a relationship that did not exist.

Cached rows store the NLI score. Their type is recomputed from lineage every
time they are loaded (and corrected in the database if it changed), so rows
written under an older rule never leak a stale type into a query.

Idempotency guarantee
---------------------
Every pair is stored in canonical order  (min(id), max(id))  so (A,B) and (B,A)
map to the same row.  The database has a UniqueConstraint("chunk_id_a","chunk_id_b"),
and we use PostgreSQL's ON CONFLICT DO NOTHING to prevent duplicates even under
concurrent requests.  The cache check also uses canonical IDs so existing pairs
are always found.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations

from sqlalchemy import tuple_ as sa_tuple
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models import ConflictPair
from services import nli
from services.retriever import CandidateChunk

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────

NLI_THRESHOLD: float = 0.70        # P(contradiction) at or above → a conflict
LEXICAL_SIM_THRESHOLD: float = 0.55  # Pair selection gate — see _lexical_cosine

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
    cached_rows = _load_cached(db, canonical_keys)
    cached_results = _reclassify_cached(db, candidate_pairs, canonical_keys, cached_rows)

    uncached_pairs = [
        (a, b)
        for (a, b), key in zip(candidate_pairs, canonical_keys)
        if key not in cached_rows
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
        for (a, b), score in zip(uncached_pairs, scores):
            conflict_type = _classify(a, b, score)
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
    all_results: list[ConflictResult] = cached_results + new_results

    elapsed = (time.monotonic() - t_start) * 1000
    log.info(
        "conflict_detector: total %.0f ms — %d cached, %d new, %d conflicts",
        elapsed, len(cached_results), len(new_results), len(all_results),
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
      • They come from DIFFERENT documents, AND
      • They share at least one named entity (CamelCase / version / ALL_CAPS),
        OR their lexical overlap exceeds LEXICAL_SIM_THRESHOLD.

    This typically cuts 190 pairs (n=20) down to 10–30.

    Why same-document pairs are excluded
    ------------------------------------
    Two passages of one document are neither a version change nor a
    disagreement between sources, and belief revision has no basis for
    preferring one over the other — there is no action such a pair can produce.

    What they do instead is corrupt the signal. NLI cross-encoders readily
    report contradiction between topically adjacent passages of ordinary prose,
    so a single long document generated seven spurious "conflicts" in testing.
    Because the reported confidence is the WORST across all pairs, those false
    positives dragged an answer with one genuine, cleanly resolved conflict down
    to "low — verify sources", and inflated the conflict count shown to the user.

    This system detects contradictions ACROSS versions and sources. Internal
    inconsistency within one document is a different problem and is out of scope.
    """
    selected = []
    for a, b in combinations(candidates, 2):
        if a.doc_id == b.doc_id:
            continue
        if _shares_entity(a.content, b.content):
            selected.append((a, b))
            continue
        if _lexical_cosine(a.content, b.content) > LEXICAL_SIM_THRESHOLD:
            selected.append((a, b))
    return selected


def _shares_entity(text_a: str, text_b: str) -> bool:
    """Return True if both texts share at least one named entity token."""
    entities_a = set(_ENTITY_RE.findall(text_a))
    entities_b = set(_ENTITY_RE.findall(text_b))
    return bool(entities_a & entities_b)


def _lexical_cosine(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between the two texts' binary bag-of-words vectors.

    For sets A and B of distinct lowercase whitespace-delimited tokens, the
    cosine of their binary indicator vectors reduces to

        |A ∩ B| / (sqrt(|A|) · sqrt(|B|))

    which is what this computes. It is a genuine cosine similarity, but over
    word PRESENCE — it ignores term frequency, word order and meaning, so it is
    much weaker than an embedding cosine. It is used only as a cheap pair
    *selection* gate before the expensive NLI step, never for scoring.

    (The module previously described this as comparing "embedding_vec"s and the
    threshold was named COSINE_SIM_THRESHOLD, both of which implied a dense
    semantic similarity that was never computed here. CandidateChunk does not
    carry embedding vectors at all.)

    Returns 0.0 when either side has no tokens.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / ((len(words_a) ** 0.5) * (len(words_b) ** 0.5))


def _load_cached(
    db: Session,
    canonical_keys: list[tuple[str, str]],
) -> dict[tuple[str, str], ConflictPair]:
    """Return the already-stored ConflictPair rows for these canonical pairs."""
    if not canonical_keys:
        return {}

    rows = (
        db.query(ConflictPair)
        .filter(
            sa_tuple(ConflictPair.chunk_id_a, ConflictPair.chunk_id_b).in_(canonical_keys)
        )
        .all()
    )
    return {(row.chunk_id_a, row.chunk_id_b): row for row in rows}


def _reclassify_cached(
    db: Session,
    pairs: list[tuple[CandidateChunk, CandidateChunk]],
    canonical_keys: list[tuple[str, str]],
    cached_rows: dict[tuple[str, str], ConflictPair],
) -> list[ConflictResult]:
    """
    Turn cached rows into ConflictResults, re-deriving each type from lineage.

    The NLI score is what the cache saves; the type is cheap to recompute and
    must follow the current rule. Rows whose stored type differs are updated so
    the Conflicts page agrees with what queries report.
    """
    results: list[ConflictResult] = []
    changed = 0
    for (a, b), key in zip(pairs, canonical_keys):
        row = cached_rows.get(key)
        if row is None:
            continue
        score = row.nli_score or 0.0
        conflict_type = _classify(a, b, score)
        if conflict_type is None:
            continue
        if row.conflict_type != conflict_type:
            row.conflict_type = conflict_type
            changed += 1
        results.append(ConflictResult(
            chunk_id_a=row.chunk_id_a,
            chunk_id_b=row.chunk_id_b,
            conflict_type=conflict_type,
            nli_score=score,
            is_cached=True,
        ))

    if changed:
        try:
            db.commit()
            log.info("conflict_detector: reclassified %d cached conflict rows.", changed)
        except Exception:
            log.error("conflict_detector: failed to update cached conflict types.", exc_info=True)
            db.rollback()
    return results


def _classify(
    a: CandidateChunk,
    b: CandidateChunk,
    nli_score: float,
) -> str | None:
    """
    Classify a pair given its NLI contradiction score.

    Returns None when the score is below NLI_THRESHOLD (not a conflict),
    "version_supersession" when both chunks belong to the same version lineage,
    and "direct_contradiction" otherwise. See "Classification" above.
    """
    if nli_score < NLI_THRESHOLD:
        return None
    if _lineage(a) == _lineage(b):
        return "version_supersession"
    return "direct_contradiction"


def _lineage(chunk: CandidateChunk) -> str:
    # A chunk without a resolved lineage is treated as its own document's
    # lineage, never as sharing an empty id with other unresolved chunks.
    return chunk.lineage_id or chunk.doc_id


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
            "conflict_id": str(uuid.uuid4()),
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

    # NOTE: this MUST be the PostgreSQL dialect's insert(). The generic
    # sqlalchemy.insert() has no .on_conflict_do_nothing(), so the previous
    # import raised AttributeError on every call — which the broad `except`
    # below swallowed into a warning. The visible symptom was that conflict
    # detection appeared to work while conflict_pairs stayed permanently empty,
    # so nothing was ever cached and the Conflicts page never showed anything.
    try:
        stmt = (
            pg_insert(ConflictPair)
            .values(rows_to_insert)
            .on_conflict_do_nothing(index_elements=["chunk_id_a", "chunk_id_b"])
        )
        db.execute(stmt)
        db.commit()
        log.info("conflict_detector: persisted %d new conflict rows.", len(rows_to_insert))
    except Exception:
        # Keep the request alive — a failed conflict write must not fail the
        # user's query — but log the traceback so this can never again look
        # like "no conflicts found".
        log.error(
            "conflict_detector: failed to persist %d conflict rows.",
            len(rows_to_insert), exc_info=True,
        )
        db.rollback()
