"""
services/version_resolver.py — Link newly ingested documents into a version lineage.

Two phases, deliberately separated:

    validate_lineage(parent_doc_id, new_version, new_date, db)
        Read-only. Checks everything that can be known before the new document
        exists: the parent is there, it is the current version of its lineage,
        the new version is strictly greater and the new date is not earlier.
        Ingestion calls this immediately after metadata validation, so an
        out-of-order version is rejected before the file is parsed, chunked and
        embedded — work that would only be thrown away by the rollback.

    apply_lineage(parent_doc_id, new_doc_id, new_version, new_date, db)
        The authoritative phase, called inside the ingestion transaction after
        the new document has been flushed. It re-loads the parent FOR UPDATE
        and runs the same guards again before superseding it: is_latest=False,
        and each of its chunks gets valid_to = new_date, is_superseded = True.

Why the guards run twice
------------------------
The early check is an optimisation and cannot be authoritative: between it and
the commit, a concurrent ingest may have superseded the same parent, which
would leave two "latest" versions in one lineage. Re-running the guards while
holding the parent's row lock closes that window — the second uploader's
re-check sees is_latest=False and is rejected. Both phases call the same
_check_guards(), so there is one implementation of the rules, run twice.

Why only the parent is touched
------------------------------
The is_latest guard means the parent is always the one version of its lineage
that is still current, so every older version was already superseded when *it*
was replaced: those documents already have is_latest=False and their chunks
already have a valid_to and is_superseded=True. Walking further up the chain
would find nothing left to change. One lineage, one current version, one
document to update.

Why the guards matter
---------------------
Query-time temporal filtering reads each chunk's validity window
[valid_from, valid_to). Those windows are only meaningful if a lineage is a
single chain whose versions follow each other in time: then exactly one version
of a lineage is valid on any given day. Branching from an older version would
leave two "latest" documents, and a version dated before its parent would give
the parent a window that ends before it starts.

    unlink_lineage(doc, db)
        The delete path. A version can only be removed if nothing points at it
        as a parent, which — because every superseded version has a child —
        means only the current version of a lineage is deletable. Removing it
        restores its parent as the current version, undoing the supersession.

lineage_roots(db, doc_ids) → {doc_id: root_doc_id}
    Identifies which lineage each document belongs to. The conflict detector
    uses it to tell a genuine version supersession (same lineage) from a
    contradiction between unrelated documents. This one *does* walk the whole
    chain, because a lineage's identity is its root, however deep it is.
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
# Every module that needs to order or match versions (lineage resolution, the
# /lineage endpoint, the reranker's version filter) MUST use version_sort_key(),
# is_version_greater() or version_matches() from this module. Do not
# re-implement version parsing anywhere else.
#
# Policy
# ------
# * A version is reduced to its sequence of integer components, in order:
#       "v1.13.1" → (1, 13, 1)      "2.2" → (2, 2)      "2024-03-01" → (2024, 3, 1)
# * Components are zero-padded to a fixed width so that trailing zeros are
#   insignificant. This makes the following hold:
#       "2.2" == "2.2.0" == "2.2.0.0"      "1" == "1.0"
# * Components compare numerically, not lexically, so "2.10" > "2.9".
# * A missing or unparseable version sorts BELOW every real version, is never
#   considered greater than anything (is_version_greater) and never matches
#   anything (version_matches).
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


def version_matches(chunk_version: Optional[str], requested: Optional[str]) -> bool:
    """
    Return True if `chunk_version` names the same release as `requested`.

    The same policy as the ordering above, so "2", "2.0" and "2.0.0" all name
    one release, while "2.1" and "2.10" do not — they are different numbers,
    and matching them as strings would serve v2.10 to someone who asked for
    v2.1. A missing or unparseable version on either side matches nothing: the
    version someone asked for has to be a version.

    This is the filter for `version` questions ("according to v2.0, …").
    """
    key = version_sort_key(requested)
    return key[0] == 1 and version_sort_key(chunk_version) == key


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_lineage(
    parent_doc_id: Optional[str],
    new_version: str,
    new_date: datetime,
    db: Session,
) -> None:
    """
    Read-only lineage guard, safe to call before the new document exists.

    Called at the very start of ingestion, right after the metadata format
    checks, so that a bad parent, a non-increasing version or a backdated
    publication date is rejected before the file is parsed, chunked and
    embedded. Nothing here writes, so there is nothing to roll back.

    This is NOT authoritative: apply_lineage() re-runs the same guards under a
    row lock inside the transaction. See the module docstring.

    Raises:
        ValueError: If parent_doc_id does not exist, or any guard fails.
    """
    if parent_doc_id is None:
        return
    _check_guards(_load_parent(db, parent_doc_id, lock=False), new_version, new_date)


def apply_lineage(
    parent_doc_id: Optional[str],
    new_doc_id: str,
    new_version: str,
    new_date: datetime,
    db: Session,
) -> LineageResult:
    """
    Supersede the parent and report how the new document joins its lineage.

    IMPORTANT: Call this *after* the new Document row has been flushed to DB
    (so it's visible to the session) but *before* committing.

    The parent is re-loaded FOR UPDATE and re-checked here, because the early
    validate_lineage() call may have been overtaken by a concurrent ingest of
    another version of the same document.

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
        ValueError: If parent_doc_id does not exist in the DB, or if any of the
                    three chain guards fails.
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
    parent = _load_parent(db, parent_doc_id, lock=True)
    _check_guards(parent, new_version, new_date)

    log.info(
        "Linking doc %s (v%s) as child of %s (v%s)",
        new_doc_id[:8], new_version,
        parent_doc_id[:8], parent.version_string,
    )

    # ── Supersede the parent ─────────────────────────────────────────────────
    parent.is_latest = False

    stamped = (
        db.query(Chunk)
        .filter(
            Chunk.doc_id == parent.doc_id,
            Chunk.is_superseded == False,  # noqa: E712
        )
        .all()
    )
    for ch in stamped:
        ch.valid_to = new_date
        ch.is_superseded = True

    superseded_chunk_count = len(stamped)
    log.info(
        "Superseded doc %s (v%s) and its %d chunks",
        parent.doc_id[:8], parent.version_string, superseded_chunk_count,
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


# ---------------------------------------------------------------------------
# The guards themselves — one implementation, run by both phases
# ---------------------------------------------------------------------------


def _load_parent(db: Session, parent_doc_id: str, *, lock: bool) -> Document:
    """
    Load the parent document, or raise ValueError if there is no such row.

    `lock=True` takes a row lock (SELECT ... FOR UPDATE) so that the guards and
    the supersession that follows them are atomic against a concurrent ingest
    of another version of the same document.
    """
    query = db.query(Document).filter(Document.doc_id == parent_doc_id)
    if lock:
        query = query.with_for_update()
    parent = query.first()
    if parent is None:
        raise ValueError(f"parent_doc_id '{parent_doc_id}' not found in the database.")
    return parent


def _check_guards(parent: Document, new_version: str, new_date: datetime) -> None:
    """Raise ValueError unless the new version may extend `parent`'s lineage."""
    # Chain guard — a lineage must stay linear (see "Why the guards matter").
    # It is also what makes the parent the only document to update: every
    # earlier version was superseded when it was itself replaced.
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


def unlink_lineage(doc: Document, db: Session) -> str:
    """
    Take `doc` out of its lineage so that it can be deleted. Returns a message.

    Deletion is only allowed for a version nothing else depends on: if another
    document names `doc` as its parent, deleting it would leave that child
    pointing at a row that no longer exists (the parent_doc_id foreign key is
    ON DELETE SET NULL, so the chain would silently break into two lineages).
    Because apply_lineage() only accepts a parent that is still is_latest,
    every superseded version already has exactly one child — so the one
    deletable version of a lineage is its current one.

    Removing the current version makes its parent current again, which is
    exactly the inverse of the supersession apply_lineage() performed: the
    parent's is_latest goes back to True and its chunks get their open-ended
    validity window back (valid_to = None, is_superseded = False). Without
    that the lineage would be left with no current version at all and every
    one of its chunks expired, so it would disappear from "current" questions
    while still sitting in the database.

    The parent is loaded FOR UPDATE, and the caller holds the same lock on
    `doc` itself, which is what serialises this against a concurrent ingest
    trying to add a new version on top of either one.

    Raises:
        ValueError: If another document depends on `doc` as its parent.
    """
    child = (
        db.query(Document)
        .filter(Document.parent_doc_id == doc.doc_id)
        .first()
    )
    if child is not None:
        raise ValueError(
            f"Document '{doc.title}' v{doc.version_string} cannot be deleted "
            f"because v{child.version_string} was published as its next "
            f"version. Delete the newer version first."
        )

    if doc.parent_doc_id is None:
        # The whole lineage was this one document; nothing else to restore.
        return "Lineage removed."

    parent = _load_parent(db, doc.parent_doc_id, lock=True)
    parent.is_latest = True

    reopened = (
        db.query(Chunk)
        .filter(
            Chunk.doc_id == parent.doc_id,
            Chunk.is_superseded == True,  # noqa: E712
        )
        .all()
    )
    for ch in reopened:
        ch.valid_to = None
        ch.is_superseded = False

    log.info(
        "Deleting doc %s (v%s); restoring parent %s (v%s) as latest, reopening %d chunks",
        doc.doc_id[:8], doc.version_string,
        parent.doc_id[:8], parent.version_string, len(reopened),
    )

    return (
        f"v{parent.version_string} is the latest version again"
        + (f"; {len(reopened)} chunk{'s' if len(reopened) != 1 else ''} reopened."
           if reopened else ".")
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
