"""
main.py — FastAPI application entry point.

Startup sequence:
  1. Load settings from .env (via config.py)
  2. Create database tables (idempotent — skips existing tables)
  3. Load embedding model (sentence-transformers)
  4. Load FAISS + BM25 indexes from disk
  5. Register all routers

Run with:
    uvicorn main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import Base, engine
from routers import health, ingest, query
from routers import conflicts as conflicts_router
from routers import analytics as analytics_router
from services import embedder, faiss_store, bm25_store, nli

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup before serving requests."""

    # 1. Create all tables (skips any that already exist)
    log.info("Running Base.metadata.create_all() …")
    import models  # noqa: F401 — import triggers model registration with Base
    Base.metadata.create_all(bind=engine)
    log.info("Database tables ready.")

    # 2. Load embedding model (downloads on first run, ~90 MB cached)
    log.info("Loading embedding model …")
    embedder.load()

    # 3. Load FAISS + BM25 indexes from disk
    log.info("Loading FAISS index …")
    faiss_store.load()
    log.info("Loading BM25 index …")
    bm25_store.load()

    # 4. Load NLI cross-encoder model (Phase 6)
    log.info("Loading NLI model …")
    try:
        nli.load()
    except Exception as exc:
        log.warning("NLI model failed to load — conflict detection will be skipped: %s", exc)

    # 5. Warn if the search indexes have drifted from the database
    _check_index_consistency()

    log.info("All services ready. ✓")
    yield  # ← server is running here

    log.info("Shutting down.")


def _check_index_consistency() -> None:
    """
    Compare the FAISS/BM25 index sizes against the chunk count in Postgres.

    The indexes are derived data; Postgres is the source of truth. They can
    drift when documents are deleted directly from the database, or when an
    older build wrote to the indexes before committing a transaction that then
    failed. Drift is silent at query time — orphaned entries are retrieved,
    fail to resolve to a row, and are dropped, so every result set quietly
    comes back short. Surfacing it at startup makes it diagnosable.
    """
    from sqlalchemy import func
    from database import SessionLocal
    from models import Chunk

    try:
        with SessionLocal() as session:
            chunk_count = session.query(func.count(Chunk.chunk_id)).scalar() or 0
    except Exception as exc:
        log.warning("Could not verify index consistency (database unreachable): %s", exc)
        return

    faiss_size = faiss_store.index_size()
    bm25_size = bm25_store.corpus_size()

    if faiss_size == chunk_count and bm25_size == chunk_count:
        log.info(
            "Index consistency OK — %d chunks in FAISS, BM25 and Postgres.",
            chunk_count,
        )
        return

    log.warning(
        "INDEX DRIFT: Postgres has %d chunks but FAISS has %d and BM25 has %d. "
        "Orphaned index entries are silently dropped during retrieval, so "
        "queries will return fewer sources than requested. "
        "Run `python scripts/rebuild_indexes.py` to rebuild both indexes "
        "from the database.",
        chunk_count, faiss_size, bm25_size,
    )


# ── App ──────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Temporal RAG with Belief Revision",
    description=(
        "A research RAG system that understands document versions, "
        "detects temporal conflicts, and applies belief revision."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Allow the Vite dev server (localhost:5173) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────

# Health check lives at /health (no /api/v1 prefix — keeps it simple)
app.include_router(health.router)

# Phase 2: ingestion + document listing
app.include_router(ingest.router, prefix="/api/v1")

# Phase 3: query + answer generation
app.include_router(query.router, prefix="/api/v1")

# Phase 6: conflict records
app.include_router(conflicts_router.router, prefix="/api/v1")

# Phase 8: analytics + timeline
app.include_router(analytics_router.router, prefix="/api/v1")
