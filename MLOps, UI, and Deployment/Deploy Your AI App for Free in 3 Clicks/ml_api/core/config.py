"""Runtime settings for the FastAPI model serving subsystem."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App-level settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="ML_API_", env_file=".env", extra="ignore")

    app_name: str = "Ames Housing ML API"
    app_version: str = "1.0.0"
    environment: str = "dev"

    model_dir: Path = Field(default=Path("outputs/api_model"))
    benchmark_dir: Path = Field(default=Path("outputs/api_benchmarks"))
    data_path: Path = Field(default=Path("data/raw/ames_housing_curated.csv"))

    joblib_artifact: str = "best_model.joblib"
    pickle_artifact: str = "best_model.pkl"
    metadata_artifact: str = "model_metadata.json"

    max_batch_size: int = 128
    max_request_bytes: int = 1_048_576
    random_seed: int = 42

    auto_train_on_startup: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings to avoid repeated env parsing."""
    return Settings()
