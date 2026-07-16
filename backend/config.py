"""
config.py — Application settings loaded from .env via Pydantic BaseSettings.
All environment variables are read once at import time and cached.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",       # project root .env
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
    llm_model: str = "llama-3.3-70b-versatile"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


# Single instance used across the application
settings = Settings()
