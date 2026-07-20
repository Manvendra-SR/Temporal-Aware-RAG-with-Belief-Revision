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

    log.info("All services ready. ✓")
    yield  # ← server is running here

    log.info("Shutting down.")


# ── App ──────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Temporal RAG with Belief Revision",
    description=(
        "A research RAG system that understands document versions, "
        "detects temporal conflicts, and applies belief revision."
    ),
    version="0.1.0",
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
