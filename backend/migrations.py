"""
migrations.py — Idempotent schema and data upgrades, run at startup after
Base.metadata.create_all().

create_all() creates missing tables but never alters existing ones, and the
project has no migration framework. The one schema change so far is small
enough to apply directly; every step here checks before it acts, so running
it on every startup is safe and a no-op once the database is current.

Conflict pairs (the resolution redesign)
----------------------------------------
1. Add conflict_pairs.preferred_chunk_id.
2. Delete pairs between two chunks of one document. The detector stopped
   producing them (see conflict_detector._select_pairs) and no rule can act
   on them, but rows written before that change would otherwise sit in the
   review queue forever.
3. Re-type every pair by lineage — the same rule the detector applies — so
   rows written under the old date-gap rule are corrected even if no query
   ever retrieves them again.
4. Convert legacy resolutions into explicit decisions, never guessing:
       temporal_preference → prefer_source, naming the later-dated chunk, but
                             only when both are dated on different days;
                             otherwise the pair is reopened for review
       scope_clarification → both_valid
       manual / anything else → reopened: it recorded no decision
   A resolution on a version change is cleared: version changes are not
   reviewed. Reviewer notes are always kept.
"""

from __future__ import annotations

import logging
from collections import Counter

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from models import (
    BOTH_VALID,
    CONTRADICTION,
    PREFER_SOURCE,
    VERSION_CHANGE,
    Chunk,
    ConflictPair,
)
from services.version_resolver import lineage_roots

log = logging.getLogger(__name__)


def upgrade(engine: Engine) -> None:
    """Bring an existing database up to the current schema and data rules."""
    _add_preferred_chunk_column(engine)
    with Session(engine) as db:
        changes = upgrade_conflict_pairs(db)
        db.commit()
    if changes:
        log.info("migrations: conflict_pairs upgraded: %s", dict(changes))


def _add_preferred_chunk_column(engine: Engine) -> None:
    columns = {c["name"] for c in inspect(engine).get_columns("conflict_pairs")}
    if "preferred_chunk_id" in columns:
        return
    ddl_type = ConflictPair.__table__.c.preferred_chunk_id.type.compile(dialect=engine.dialect)
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE conflict_pairs ADD COLUMN preferred_chunk_id {ddl_type}"))
    log.info("migrations: added conflict_pairs.preferred_chunk_id")


def upgrade_conflict_pairs(db: Session) -> Counter:
    """Apply steps 2–4 of the module docstring. Returns what changed; does not commit."""
    changes: Counter = Counter()
    rows = db.query(ConflictPair).all()
    if not rows:
        return changes

    ids = {cid for r in rows for cid in (r.chunk_id_a, r.chunk_id_b)}
    chunks = {c.chunk_id: c for c in db.query(Chunk).filter(Chunk.chunk_id.in_(ids)).all()}
    roots = lineage_roots(db, {c.doc_id for c in chunks.values()})

    for row in rows:
        a, b = chunks.get(row.chunk_id_a), chunks.get(row.chunk_id_b)
        if a is None or b is None:
            continue   # the chunks' ON DELETE CASCADE normally removes these

        if a.doc_id == b.doc_id:
            db.delete(row)
            changes["same_document_deleted"] += 1
            continue

        kind = VERSION_CHANGE if roots[a.doc_id] == roots[b.doc_id] else CONTRADICTION
        if row.conflict_type != kind:
            row.conflict_type = kind
            changes["retyped"] += 1

        if row.has_valid_resolution():
            continue
        if kind == CONTRADICTION and row.is_resolved:
            _convert_legacy_resolution(row, a, b, changes)
        elif row.is_resolved or row.resolution_type or row.preferred_chunk_id:
            _reopen(row)
            changes["resolution_cleared"] += 1

    return changes


def _convert_legacy_resolution(row: ConflictPair, a: Chunk, b: Chunk, changes: Counter) -> None:
    legacy = row.resolution_type
    if legacy == "scope_clarification":
        row.resolution_type, row.preferred_chunk_id = BOTH_VALID, None
        changes["scope_clarification_to_both_valid"] += 1
        return

    if legacy == "temporal_preference":
        day_a = a.valid_from.date() if a.valid_from else None
        day_b = b.valid_from.date() if b.valid_from else None
        if day_a is not None and day_b is not None and day_a != day_b:
            row.resolution_type = PREFER_SOURCE
            row.preferred_chunk_id = a.chunk_id if day_a > day_b else b.chunk_id
            changes["temporal_preference_to_prefer_source"] += 1
            return

    # manual, unknown, or a temporal_preference whose "newer" side is undefined.
    _reopen(row)
    changes[f"{legacy or 'untyped'}_reopened"] += 1


def _reopen(row: ConflictPair) -> None:
    """Back to pending. The reviewer's note is kept so the next reviewer sees it."""
    row.is_resolved = False
    row.resolution_type = None
    row.preferred_chunk_id = None
    row.resolved_at = None
