"""Environment settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global environment-configurable settings."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PEFT_PLATFORM_", extra="ignore")

    mlflow_tracking_uri: str = Field(default="sqlite:///artifacts/mlflow/mlflow.db")
    mlflow_artifacts_root: str = Field(default="artifacts/mlflow")
    data_root: str = Field(default="data")
    artifacts_root: str = Field(default="artifacts")
    default_device: str = Field(default="auto")
