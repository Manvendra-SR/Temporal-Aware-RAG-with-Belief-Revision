"""
config.py — Application settings loaded from .env via Pydantic BaseSettings.
All environment variables are read once at import time and cached.

This module is the single source of truth for tunable runtime configuration.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory containing backend/ and frontend/.
# Resolved from __file__ so the app works regardless of the current
# working directory (previously a relative "../.env" only worked when
# uvicorn was launched from inside backend/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    # Where the FAISS and BM25 index files live. Empty means <project>/data.
    # Set it (with DB_NAME) to run against a separate corpus — the evaluation in
    # evaluation/README.md does this so its documents never mix with yours.
    data_dir: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    # SQLAlchemy engine echo. Defaults to False — echoing every statement made
    # the development logs unreadable and hid application-level log lines.
    db_echo: bool = False


# Single instance used across the application
settings = Settings()

# Search-index directory. Resolved here, after settings, so DATA_DIR can be set
# in the environment or .env like every other setting. A relative path is
# relative to the process's working directory.
DATA_DIR = Path(settings.data_dir).resolve() if settings.data_dir else PROJECT_ROOT / "data"
