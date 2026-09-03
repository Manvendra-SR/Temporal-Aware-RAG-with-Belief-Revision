"""
tests/conftest.py — shared test fixtures and import path setup.

These tests deliberately cover only pure, deterministic logic: version
ordering, query analysis, temporal scoring, belief revision and the conflict
detector's helper functions. None of them touch PostgreSQL, FAISS, the
embedding model or the LLM, so the suite runs in under a second with no
services running and no network access.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Make `backend/` importable so tests can `from services import ...` exactly
# the way the application does.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.retriever import CandidateChunk  # noqa: E402


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def days_before(days: int, reference: datetime = NOW) -> datetime:
    """A UTC datetime `days` before the fixed reference time."""
    return reference - timedelta(days=days)


def make_chunk(
    chunk_id: str = "chunk-a",
    *,
    doc_id: str = "doc-1",
    doc_title: str = "Test Doc",
    content: str = "some content",
    relevance_score: float = 0.5,
    semantic_score: float = 0.5,
    bm25_score: float = 1.0,
    rrf_score: float = 0.03,
    valid_from: datetime | None = None,
    version_string: str | None = None,
    is_latest: bool = True,
    is_superseded: bool = False,
) -> CandidateChunk:
    """Build a CandidateChunk with sensible defaults for tests."""
    return CandidateChunk(
        chunk_id=chunk_id,
        faiss_id=0,
        doc_id=doc_id,
        doc_title=doc_title,
        content=content,
        content_snippet=content[:200],
        section_heading=None,
        bm25_score=bm25_score,
        semantic_score=semantic_score,
        rrf_score=rrf_score,
        relevance_score=relevance_score,
        valid_from=valid_from,
        version_string=version_string,
        is_latest=is_latest,
        is_superseded=is_superseded,
    )


class FakeQuery:
    """Minimal stand-in for a SQLAlchemy Query that always finds nothing."""

    def filter(self, *_args, **_kwargs) -> "FakeQuery":
        return self

    def first(self):
        return None


class FakeSession:
    """
    Minimal stand-in for a SQLAlchemy Session.

    belief_revision.revise() only uses the session to look up stored conflict
    resolutions. `stored` maps a (chunk_id_a, chunk_id_b) tuple to the
    ConflictPair-like object that lookup should return.
    """

    def __init__(self, stored: dict | None = None):
        self._stored = stored or {}
        self._filters: list = []

    def query(self, *_args):
        if not self._stored:
            return FakeQuery()
        return _StoredQuery(self._stored)


class _StoredQuery:
    def __init__(self, stored: dict):
        self._stored = stored
        self._ids: list[str] = []

    def filter(self, *criteria):
        # Pull the right-hand literal out of each `Column == value` clause.
        for clause in criteria:
            try:
                self._ids.append(clause.right.value)
            except AttributeError:
                continue
        return self

    def first(self):
        key = tuple(self._ids[:2])
        return self._stored.get(key)


@pytest.fixture
def session_without_resolutions() -> FakeSession:
    return FakeSession()
