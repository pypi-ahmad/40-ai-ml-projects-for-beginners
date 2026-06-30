"""Application configuration via pydantic-settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


class Settings(BaseSettings):
    """Typed environment settings for API runtime behavior."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core app settings
    app_name: str = "ML Model API Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging and observability
    log_level: str = "INFO"
    log_format: str = "text"
    request_log_sample_rate: float = 1.0

    # API access/security
    cors_origins: str = "*"
    rate_limit_per_minute: int = 120
    api_key: str = ""
    max_batch_size: int = 128
    max_request_body_bytes: int = 1_000_000

    # Model + metadata artifacts
    model_path: Path = PROJECT_ROOT / "models" / "model.joblib"
    metadata_path: Path = PROJECT_ROOT / "models" / "metadata.json"
    feature_schema_version: str = "california-housing-v1"

    # Explainability runtime guardrails
    explain_max_background_rows: int = 128

    # SQLite-backed API statistics
    metrics_db_path: Path = PROJECT_ROOT / "artifacts" / "performance" / "api_metrics.db"

    # Optional experiment tracking
    wandb_project: str = "ml-api-deployment"
    wandb_entity: str = ""

    # Optional admin behavior
    enable_reload_endpoint: bool = True

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if v not in ("text", "json"):
            raise ValueError('log_format must be "text" or "json"')
        return v

    @field_validator("rate_limit_per_minute")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")
        return v

    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_batch_size must be >= 1")
        if v > 5000:
            raise ValueError("max_batch_size must be <= 5000")
        return v

    @field_validator("max_request_body_bytes")
    @classmethod
    def validate_max_request_body_bytes(cls, v: int) -> int:
        if v < 1024:
            raise ValueError("max_request_body_bytes must be >= 1024")
        return v

    @field_validator("request_log_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("request_log_sample_rate must be in (0, 1].")
        return v

    @field_validator("explain_max_background_rows")
    @classmethod
    def validate_explain_rows(cls, v: int) -> int:
        if v < 1:
            raise ValueError("explain_max_background_rows must be >= 1")
        if v > 5000:
            raise ValueError("explain_max_background_rows must be <= 5000")
        return v


settings = Settings()
