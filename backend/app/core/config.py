"""
backend/app/core/config.py
--------------------------
Application-wide settings loaded from environment variables / .env file.
Uses pydantic-settings for type-safe configuration management.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the GenAI Data Analyst backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_NAME: str = "GenAI Data Analyst"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    API_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------ #
    # CORS — accepts a comma-separated string or list from .env
    # ------------------------------------------------------------------ #
    ALLOWED_ORIGINS: list[str] | str = ["http://localhost:3000"]

    @field_validator("ALLOWED_ORIGINS", mode="after")
    @classmethod
    def _parse_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                import json
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # ------------------------------------------------------------------ #
    # File Upload
    # ------------------------------------------------------------------ #
    MAX_UPLOAD_SIZE_MB: int = 50

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    ALLOWED_EXTENSIONS: list[str] = [".csv", ".xlsx", ".xls"]

    # ------------------------------------------------------------------ #
    # Generative AI (Phase 5)
    # ------------------------------------------------------------------ #
    GEMINI_API_KEY: str = ""


# Singleton — import this throughout the application
settings = Settings()

