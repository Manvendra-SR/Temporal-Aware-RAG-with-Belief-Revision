"""
main.py — FastAPI application entry point.

Startup sequence:
  1. Load settings from .env (via config.py)
  2. Create database tables (idempotent — skips existing tables)
  3. Seed domain_config defaults
  4. Load embedding model (sentence-transformers)
  5. Load FAISS + BM25 indexes from disk
  6. Register all routers

Run with:
    uvicorn main:app --reload
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import Base, SessionLocal, engine
from routers import health, ingest, query
from services import embedder, faiss_store, bm25_store

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

    # 2. Seed DomainConfig defaults (safe to run on every restart)
    _seed_domain_config()

    # 3. Load embedding model (downloads on first run, ~90 MB cached)
    log.info("Loading embedding model …")
    embedder.load()

    # 4. Load FAISS + BM25 indexes from disk
    log.info("Loading FAISS index …")
    faiss_store.load()
    log.info("Loading BM25 index …")
    bm25_store.load()

    log.info("All services ready. ✓")
    yield  # ← server is running here

    log.info("Shutting down.")


def _seed_domain_config() -> None:
    """Inserts default domain half-life entries if they don't already exist."""
    from models import DomainConfig

    defaults = [
        ("npm_docs", 30, "npm package documentation"),
        ("python_docs", 90, "Python language documentation"),
        ("arxiv_cs", 180, "arXiv computer science papers"),
        ("pytorch_docs", 90, "PyTorch framework documentation"),
        ("legal", 730, "Legal documents and regulations"),
        ("general", 365, "General-purpose documents"),
    ]

    db = SessionLocal()
    try:
        for domain, days, description in defaults:
            existing = db.get(DomainConfig, domain)
            if existing is None:
                db.add(
                    DomainConfig(
                        domain=domain,
                        half_life_days=days,
                        description=description,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
        db.commit()
        log.info("DomainConfig seeded with %d domains.", len(defaults))
    except Exception as exc:
        log.warning("Could not seed DomainConfig: %s", exc)
        db.rollback()
    finally:
        db.close()


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
