"""
services/version_resolver.py — Link newly ingested documents into a version lineage.

resolve(parent_doc_id, new_doc_id, new_version, new_date, db) → LineageResult

Algorithm:
    1. If parent_doc_id is None → new lineage, mark new doc is_latest=True, done.
    2. Load parent from DB. If not found → raise ValueError (caller returns 422).
    3. Guards (each raises ValueError):
         - the parent must be the latest version of its lineage
         - new_version must be strictly greater than parent.version_string
         - new_date must not be earlier than the parent's published_at
    4. Walk the lineage from parent upward to mark ALL ancestors is_latest=False.
    5. Stamp all active (non-superseded) chunks of older docs:
         valid_to = new_date, is_superseded = True
    6. Set new doc is_latest=True, parent_doc_id = parent.doc_id.
    7. Return LineageResult.

Why the guards matter
---------------------
Query-time temporal filtering reads each chunk's validity window
[valid_from, valid_to). Those windows are only meaningful if a lineage is a
single chain whose versions follow each other in time: then exactly one version
of a lineage is valid on any given day. Branching from an older version would
leave two "latest" documents, and a version dated before its parent would give
the parent a window that ends before it starts.

lineage_roots(db, doc_ids) → {doc_id: root_doc_id}
    Identifies which lineage each document belongs to. The conflict detector
    uses it to tell a genuine version supersession (same lineage) from a
    contradiction between unrelated documents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

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
# Version comparison — the single ordering policy for the whole application
# ---------------------------------------------------------------------------
#
# Every module that needs to order versions (lineage resolution, the /lineage
# endpoint, the reranker's version matching) MUST use version_sort_key() /
# is_version_greater() from this module. Do not re-implement version parsing.
#
# Policy
# ------
# * A version is reduced to its sequence of integer components, in order:
#       "v1.13.1" → (1, 13, 1)      "2.2" → (2, 2)      "2024-03-01" → (2024, 3, 1)
# * Components are zero-padded to a fixed width so that trailing zeros are
#   insignificant. This makes the following hold:
#       "2.2" == "2.2.0" == "2.2.0.0"      "1" == "1.0"
# * Components compare numerically, not lexically, so "2.10" > "2.9".
# * A missing or unparseable version sorts BELOW every real version, and is
#   never considered greater than anything (see is_version_greater).
#
# Note: because components are compared positionally, a date-style version
# ("2024-03-01") and a semver-style version ("2.2") are not meaningfully
# comparable — a date always sorts higher. A single lineage must therefore use
# one scheme consistently. This is documented in the README.

_VERSION_COMPONENTS = 6   # components kept/compared; extra components are ignored

_DIGIT_RUN_RE = re.compile(r"\d+")


def version_sort_key(version_str: Optional[str]) -> tuple:
    """
    Return a tuple that sorts versions correctly, for use as a `key=` function.

    The first element is a "has a parseable version" flag, so missing and
    malformed versions always sort below real ones. The remaining elements are
    the zero-padded numeric components.

    Examples:
        version_sort_key("2.2")     == version_sort_key("2.2.0")
        version_sort_key("2.10")     > version_sort_key("2.9")
        version_sort_key(None)       < version_sort_key("0.1")
        version_sort_key("bad!!")    == version_sort_key(None)
    """
    if not version_str or not version_str.strip():
        return (0,) + (0,) * _VERSION_COMPONENTS

    nums = _DIGIT_RUN_RE.findall(version_str)
    if not nums:
        # Malformed — no digits at all. Treated exactly like a missing version.
        return (0,) + (0,) * _VERSION_COMPONENTS

    components = [int(n) for n in nums[:_VERSION_COMPONENTS]]
    components += [0] * (_VERSION_COMPONENTS - len(components))
    return (1,) + tuple(components)


def is_version_greater(a: Optional[str], b: Optional[str]) -> bool:
    """
    Return True if version `a` is strictly greater than version `b`.

    A missing/malformed `a` is never greater than anything (not even another
    missing version), which is what makes the lineage guard reject uploads that
    omit a usable version.
    """
    key_a = version_sort_key(a)
    if key_a[0] == 0:
        return False
    return key_a > version_sort_key(b)


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

    # Chain guard — a lineage must stay linear (see "Why the guards matter").
    if not parent.is_latest:
        raise ValueError(
            f"Document '{parent.title}' v{parent.version_string} is not the latest "
            f"version of its lineage. Upload the new version as a child of the "
            f"latest version instead."
        )

    # Version ordering guard — reject if new version is not strictly greater.
    # Uses the shared ordering policy so "2.2" is correctly rejected against a
    # "2.2.0" parent (they are equal) and "2.10" is correctly accepted over "2.9".
    if parent.version_string and not is_version_greater(new_version, parent.version_string):
        raise ValueError(
            f"New version '{new_version}' must be strictly greater than "
            f"the parent version '{parent.version_string}'."
        )

    # Date ordering guard — the parent's validity window ends at new_date, so
    # new_date before the parent's own start would make that window empty.
    if parent.published_at is not None and _as_utc(new_date) < _as_utc(parent.published_at):
        raise ValueError(
            f"New version's publication date {new_date.date().isoformat()} is "
            f"earlier than the parent version's "
            f"({parent.published_at.date().isoformat()})."
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


def lineage_roots(db: Session, doc_ids: Iterable[str]) -> dict[str, str]:
    """
    Map each document id to the id of the root document of its lineage.

    A document with no parent is its own root. Walks parent links one level
    per query for all documents at once, so the cost is one query per level of
    the deepest lineage involved, not one per document.
    """
    doc_ids = list(doc_ids)
    parent_of: dict[str, Optional[str]] = {}
    pending = set(doc_ids)
    while pending:
        rows = (
            db.query(Document.doc_id, Document.parent_doc_id)
            .filter(Document.doc_id.in_(pending))
            .all()
        )
        for row in rows:
            parent_of[row.doc_id] = row.parent_doc_id
        # Unknown ids are treated as roots rather than looked up forever.
        for missing in pending - {row.doc_id for row in rows}:
            parent_of[missing] = None
        pending = {p for p in parent_of.values() if p and p not in parent_of}

    roots: dict[str, str] = {}
    for doc_id in doc_ids:
        current, seen = doc_id, set()
        while parent_of.get(current) and current not in seen:
            seen.add(current)
            current = parent_of[current]
        roots[doc_id] = current
    return roots


def _as_utc(d: datetime) -> datetime:
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)
