"""Runtime configuration and environment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_sql_assistant.constants import (
    APP_STATE_DB_PATH,
    COMPARISON_MODEL,
    DEFAULT_MODEL,
    JUDGE_MODEL,
    NORTHWIND_DB_PATH,
    NORTHWIND_SCALED_DB_PATH,
    OLLAMA_HOST,
)


class ModelConfig(BaseModel):
    """Model selection and generation controls."""

    generator_model: str = DEFAULT_MODEL
    comparison_model: str = COMPARISON_MODEL
    judge_model: str = JUDGE_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    num_predict: int = Field(default=256, ge=32, le=2048)
    timeout_seconds: int = Field(default=300, ge=5, le=1200)
    request_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.1, le=30.0)


class DatabaseConfig(BaseModel):
    """Database and app state paths."""

    raw_db_path: Path = NORTHWIND_DB_PATH
    scaled_db_path: Path = NORTHWIND_SCALED_DB_PATH
    app_state_db_path: Path = APP_STATE_DB_PATH
    active_db: Literal["raw", "scaled"] = "scaled"

    @property
    def active_db_path(self) -> Path:
        return self.scaled_db_path if self.active_db == "scaled" else self.raw_db_path


class SafetyConfig(BaseModel):
    """Safety and query limits."""

    max_rows: int = Field(default=5_000, ge=1, le=100_000)
    max_query_seconds: float = Field(default=30.0, ge=0.1, le=300.0)
    allow_multi_statement: bool = False
    strict_join_checks: bool = True


class EvaluationConfig(BaseModel):
    """Benchmark execution controls."""

    benchmark_cases_file: Path = Path("benchmarks/benchmark_cases.json")
    full_matrix: bool = True


class AppSettings(BaseSettings):
    """Settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_prefix="AI_SQL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: HttpUrl = OLLAMA_HOST
    models: ModelConfig = ModelConfig()
    database: DatabaseConfig = DatabaseConfig()
    safety: SafetyConfig = SafetyConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def get_settings() -> AppSettings:
    """Load settings once per call site.

    Returns:
        AppSettings: Parsed runtime settings.
    """

    return AppSettings()


__all__ = [
    "AppSettings",
    "DatabaseConfig",
    "EvaluationConfig",
    "ModelConfig",
    "SafetyConfig",
    "get_settings",
]
