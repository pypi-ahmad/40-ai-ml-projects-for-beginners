"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_spreadsheet_analytics.constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SCOPES,
    DEFAULT_STATE_DB,
)


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    google_service_account_json: Path | None = Field(default=None, alias="GOOGLE_SERVICE_ACCOUNT_JSON")
    google_oauth_client_secret_json: Path | None = Field(default=None, alias="GOOGLE_OAUTH_CLIENT_SECRET_JSON")
    google_scopes: str = Field(default=",".join(DEFAULT_SCOPES), alias="GOOGLE_SCOPES")

    default_spreadsheet_ids: str = Field(default="", alias="DEFAULT_SPREADSHEET_IDS")
    default_worksheets: str = Field(default="", alias="DEFAULT_WORKSHEETS")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_primary_model: str = Field(default="qwen3.5:4b", alias="OLLAMA_PRIMARY_MODEL")
    ollama_secondary_model: str = Field(default="granite4.1:3b", alias="OLLAMA_SECONDARY_MODEL")
    ollama_judge_model: str = Field(default="granite4.1:3b", alias="OLLAMA_JUDGE_MODEL")
    ollama_temperature: float = Field(default=0.0, alias="OLLAMA_TEMPERATURE")

    state_db_path: Path = Field(default=DEFAULT_STATE_DB, alias="STATE_DB_PATH")
    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR, alias="CACHE_DIR")
    report_dir: Path = Field(default=DEFAULT_REPORT_DIR, alias="REPORT_DIR")
    benchmark_cases_path: Path = Field(
        default=Path("data/benchmarks/questions.json"), alias="BENCHMARK_CASES_PATH"
    )

    @field_validator("ollama_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("Business analytics mode requires deterministic decoding: OLLAMA_TEMPERATURE must be 0")
        return value

    @field_validator("cache_dir", "report_dir", "state_db_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @property
    def scopes(self) -> list[str]:
        return [scope.strip() for scope in self.google_scopes.split(",") if scope.strip()]

    @property
    def spreadsheet_ids(self) -> list[str]:
        return [item.strip() for item in self.default_spreadsheet_ids.split(",") if item.strip()]

    @property
    def worksheet_names(self) -> list[str]:
        return [item.strip() for item in self.default_worksheets.split(",") if item.strip()]

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
