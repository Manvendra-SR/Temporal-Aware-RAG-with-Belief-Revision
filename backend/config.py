"""
config.py — Application settings loaded from .env via Pydantic BaseSettings.
All environment variables are read once at import time and cached.

This module is the single source of truth for tunable runtime configuration,
including the temporal decay half-life used by the reranker.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory containing backend/ and frontend/.
# Resolved from __file__ so the app works regardless of the current
# working directory (previously a relative "../.env" only worked when
# uvicorn was launched from inside backend/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    db_user: str = "postgres"
    db_password: str = "password"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "temporal_rag"

    @property
    def database_url(self) -> str:
        from sqlalchemy import URL
        url = URL.create(
            drivername="postgresql",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name
        )
        return url.render_as_string(hide_password=False)

    # LLM — Groq
    groq_api_key: str = ""
    # Groq retires hosted models fairly often; the previous default
    # ("llama-3.3-70b-versatile") had been decommissioned, so every answer
    # request failed with a 404 while retrieval appeared to work fine.
    # Run `python scripts/list_llm_models.py` to see what your key can reach.
    llm_model: str = "openai/gpt-oss-120b"

    # Embeddings
    # "auto" picks CUDA when available and falls back to CPU. Set explicitly
    # ("cpu" / "cuda") to pin the device. Hardcoding "cuda" previously made the
    # app unstartable on any machine without a GPU.
    embedding_device: str = "auto"

    # ── Temporal decay ───────────────────────────────────────────────────────
    # SINGLE SOURCE OF TRUTH for temporal decay. Every module that needs a
    # half-life reads it from here (services/temporal_reranker.py).
    #
    # A chunk's temporal weight is 2 ** (-age_days / temporal_half_life_days),
    # so a chunk exactly this many days old scores 0.5.
    #
    # This is deliberately a single GLOBAL value. The original design sketched a
    # per-domain `domain_config` table; that was never implemented and the
    # `domain` concept does not exist in the data model, so per-domain half-lives
    # are not supported. See README "Temporal decay".
    temporal_half_life_days: int = 180

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    # SQLAlchemy engine echo. Defaults to False — echoing every statement made
    # the development logs unreadable and hid application-level log lines.
    db_echo: bool = False


# Single instance used across the application
settings = Settings()
