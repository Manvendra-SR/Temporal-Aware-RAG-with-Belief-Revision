"""
services/version_resolver.py — Link newly ingested documents into a version lineage.

resolve(parent_doc_id, new_doc_id, new_version, new_date, db) → LineageResult

Algorithm:
    1. If parent_doc_id is None → new lineage, mark new doc is_latest=True, done.
    2. Load parent from DB. If not found → raise ValueError (caller returns 422).
    3. Guard: new_version must be strictly greater than parent.version_string.
    4. Walk the lineage from parent upward to mark ALL ancestors is_latest=False.
    5. Stamp all active (non-superseded) chunks of older docs:
         valid_to = new_date, is_superseded = True
    6. Set new doc is_latest=True, parent_doc_id = parent.doc_id.
    7. Return LineageResult.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from models import Chunk, Document

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LineageResult:
    parent_doc_id: Optional[str]
    superseded_chunk_count: int
    is_new_lineage: bool        # True if this is the first doc in this lineage
    lineage_message: str        # Human-readable summary for the API response


# ---------------------------------------------------------------------------
# Version comparison helpers
# ---------------------------------------------------------------------------


def _parse_version(version_str: Optional[str]) -> tuple[tuple, str]:
    """
    Parse a version string into a sortable tuple and canonical string.
    Returns (sort_key_tuple, canonical_str).

    Handles:
      "2.2"      → ((2, 2, 0, 0), "2.2")
      "1.13.1"   → ((1, 13, 1, 0), "1.13.1")
      "2024-03"  → ((2024, 3, 0, 0), "2024-03")
      None       → ((0,), "")
    """
    if not version_str:
        return (0,), ""

    # Try date-style: YYYY-MM or YYYY-MM-DD
    date_m = re.match(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$", version_str)
    if date_m:
        parts = [int(x) for x in date_m.groups() if x is not None]
        return tuple(parts), version_str  # type: ignore[return-value]

    # Try numeric semver: 1.2.3.4
    nums = re.findall(r"\d+", version_str)
    if nums:
        return tuple(int(n) for n in nums), version_str  # type: ignore[return-value]

    return (0,), version_str


def _version_gt(a: Optional[str], b: Optional[str]) -> bool:
    """Return True if version a is strictly greater than version b."""
    ka, _ = _parse_version(a)
    kb, _ = _parse_version(b)
    return ka > kb


def _version_eq(a: Optional[str], b: Optional[str]) -> bool:
    ka, _ = _parse_version(a)
    kb, _ = _parse_version(b)
    return ka == kb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(
    parent_doc_id: Optional[str],
    new_doc_id: str,
    new_version: str,
    new_date: datetime,
    db: Session,
) -> LineageResult:
    """
    Resolve version lineage for the newly created document.

    IMPORTANT: Call this *after* the new Document row has been flushed to DB
    (so it's visible to the session) but *before* committing.

    Args:
        parent_doc_id: The UUID of the explicit parent document, or None for
                       a brand-new lineage.
        new_doc_id:    The UUID of the newly inserted document.
        new_version:   The version string of the new document (already validated).
        new_date:      The published_at datetime of the new document (UTC).
        db:            Active SQLAlchemy session.

    Returns:
        LineageResult with lineage details.

    Raises:
        ValueError: If parent_doc_id does not exist in the DB, or if
                    new_version is not strictly greater than the parent's version.
    """
    # ── Case 1: Brand-new lineage ────────────────────────────────────────────
    if parent_doc_id is None:
        log.info("New lineage started for doc_id=%s version=%s", new_doc_id[:8], new_version)
        return LineageResult(
            parent_doc_id=None,
            superseded_chunk_count=0,
            is_new_lineage=True,
            lineage_message="New document lineage started.",
        )

    # ── Case 2: New version of an existing document ──────────────────────────
    parent = db.query(Document).filter(Document.doc_id == parent_doc_id).first()
    if parent is None:
        raise ValueError(f"parent_doc_id '{parent_doc_id}' not found in the database.")

    # Version ordering guard — reject if new version is not strictly greater
    if parent.version_string and not _version_gt(new_version, parent.version_string):
        raise ValueError(
            f"New version '{new_version}' must be strictly greater than "
            f"the parent version '{parent.version_string}'."
        )

    log.info(
        "Linking doc %s (v%s) as child of %s (v%s)",
        new_doc_id[:8], new_version,
        parent_doc_id[:8], parent.version_string,
    )

    # ── Walk the full ancestor chain, marking all as not-latest ─────────────
    # We walk up from parent to root to collect all ancestors, then mark them.
    ancestors: list[Document] = []
    current = parent
    visited: set[str] = set()
    while current is not None and current.doc_id not in visited:
        visited.add(current.doc_id)
        ancestors.append(current)
        if current.parent_doc_id:
            current = db.query(Document).filter(Document.doc_id == current.parent_doc_id).first()
        else:
            break

    superseded_chunk_count = 0

    for ancestor in ancestors:
        if ancestor.is_latest:
            ancestor.is_latest = False
            log.info(
                "Marked doc %s (v%s) is_latest=False",
                ancestor.doc_id[:8], ancestor.version_string,
            )

        # Stamp all active chunks of this ancestor
        chunks_to_stamp = (
            db.query(Chunk)
            .filter(
                Chunk.doc_id == ancestor.doc_id,
                Chunk.is_superseded == False,  # noqa: E712
            )
            .all()
        )
        for ch in chunks_to_stamp:
            ch.valid_to = new_date
            ch.is_superseded = True
            superseded_chunk_count += 1

    log.info(
        "Superseded %d chunks across %d ancestor docs",
        superseded_chunk_count, len(ancestors),
    )

    # ── Build lineage message ─────────────────────────────────────────────────
    parts = []
    if parent.version_string:
        parts.append(f"Supersedes v{parent.version_string}")
    if superseded_chunk_count:
        parts.append(f"{superseded_chunk_count} chunks marked expired")
    lineage_message = ". ".join(parts) + "." if parts else "Linked to lineage."

    return LineageResult(
        parent_doc_id=parent_doc_id,
        superseded_chunk_count=superseded_chunk_count,
        is_new_lineage=False,
        lineage_message=lineage_message,
    )
