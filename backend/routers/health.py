"""
routers/health.py — GET /health endpoint.

Returns the application status and whether the database is reachable.
Used by the frontend sidebar to display the Connected / Disconnected indicator.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    faiss_index_size: int
    bm25_corpus_size: int


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Returns service status including DB connectivity and index sizes.
    The endpoint always returns HTTP 200 so the frontend can surface issues.
    """
    from services import faiss_store, bm25_store

    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return HealthResponse(
        status="ok",
        db_connected=db_connected,
        faiss_index_size=faiss_store.index_size(),
        bm25_corpus_size=bm25_store.corpus_size(),
    )
