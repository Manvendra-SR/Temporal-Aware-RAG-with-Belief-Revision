"""
tests/conftest.py — shared test fixtures and import path setup.

These tests deliberately cover only pure, deterministic logic: version
ordering, query analysis, temporal filtering and ranking, belief revision and
the conflict detector's helper functions. None of them touch PostgreSQL, FAISS, the
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

from services.query_analyzer import QueryAnalysis, TemporalIntent  # noqa: E402
from services.retriever import CandidateChunk  # noqa: E402


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def days_before(days: int, reference: datetime = NOW) -> datetime:
    """A UTC datetime `days` before the fixed reference time."""
    return reference - timedelta(days=days)


# Interpretations built directly, so tests of downstream stages never depend on
# the LLM-based analyzer (which is tested separately in test_query_analyzer.py).
CURRENT = QueryAnalysis(intent=TemporalIntent.CURRENT, source="llm")
HISTORICAL = QueryAnalysis(intent=TemporalIntent.HISTORICAL, source="llm")


def pinned(version: str) -> QueryAnalysis:
    """An analysis of a question that names `version`."""
    return QueryAnalysis(intent=TemporalIntent.VERSION, version=version, source="llm")


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
    valid_to: datetime | None = None,
    version_string: str | None = None,
    is_latest: bool = True,
    is_superseded: bool = False,
    lineage_id: str = "",
) -> CandidateChunk:
    """Build a CandidateChunk with sensible defaults for tests."""
    return CandidateChunk(
        chunk_id=chunk_id,
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
        valid_to=valid_to,
        version_string=version_string,
        is_latest=is_latest,
        is_superseded=is_superseded,
        lineage_id=lineage_id or doc_id,
    )


# ── The running example: one document lineage with three versions ────────────
#
#   v1.0  2022-01-10 → 2023-03-15   "Rahul is the CEO"
#   v2.0  2023-03-15 → 2025-02-01   "Rahul resigned in March 2023. Priya became CEO"
#   v3.0  2025-02-01 → (current)    "Priya continues as CEO"
#
# plus an UNRELATED document (its own lineage) that contradicts v3.0:
#
#   press 2025-06-01 → (current)    "Arjun is the CEO"
#
# The reference "today" for all of these is NOW (2026-01-01).

def utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


V1_DATE, V2_DATE, V3_DATE, PRESS_DATE = utc(2022, 1, 10), utc(2023, 3, 15), utc(2025, 2, 1), utc(2025, 6, 1)


def ceo_versions(*, rrf: tuple[float, float, float] = (0.030, 0.025, 0.020)) -> list[CandidateChunk]:
    """
    The three versions of the handbook, as retrieval would return them.

    Default RRF scores rank the OLDEST version highest — "Rahul is the CEO" is
    the cleanest textual match for "who is the CEO?" — which is exactly the
    situation plain RAG gets wrong.
    """
    common = dict(doc_title="ABC Handbook", lineage_id="handbook")
    return [
        make_chunk("v1", doc_id="doc-v1", content="Rahul is the CEO of ABC Ltd.",
                   valid_from=V1_DATE, valid_to=V2_DATE, version_string="1.0",
                   is_latest=False, is_superseded=True, rrf_score=rrf[0], **common),
        make_chunk("v2", doc_id="doc-v2",
                   content="Rahul resigned in March 2023. Priya became the CEO of ABC Ltd.",
                   valid_from=V2_DATE, valid_to=V3_DATE, version_string="2.0",
                   is_latest=False, is_superseded=True, rrf_score=rrf[1], **common),
        make_chunk("v3", doc_id="doc-v3", content="Priya continues as the CEO of ABC Ltd.",
                   valid_from=V3_DATE, version_string="3.0", rrf_score=rrf[2], **common),
    ]


def press_release(rrf: float = 0.015) -> CandidateChunk:
    """An unrelated document that contradicts the current handbook."""
    return make_chunk("press", doc_id="doc-press", doc_title="Press Release",
                      content="Arjun is the CEO of ABC Ltd.", valid_from=PRESS_DATE,
                      version_string="1.0", rrf_score=rrf, lineage_id="press")


def analysis(intent: str, **fields) -> QueryAnalysis:
    """A QueryAnalysis for `intent`, e.g. analysis("range", start_date=date(...))."""
    return QueryAnalysis(intent=TemporalIntent(intent), source="llm", **fields)


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
