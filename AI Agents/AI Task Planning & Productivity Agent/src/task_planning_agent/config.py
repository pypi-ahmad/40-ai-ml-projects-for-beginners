"""Runtime configuration for the productivity agent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    """Environment-backed runtime settings."""

    model_config = SettingsConfigDict(env_prefix="TPA_", env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"
    ollama_base_url: str = "http://localhost:11434"
    jwt_secret: str = "change-me-in-production"


class AppConfig(BaseModel):
    """Top-level app config loaded from YAML."""

    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def planner(self) -> dict[str, Any]:
        return self.raw.get("planner", {})

    @property
    def scheduling(self) -> dict[str, Any]:
        return self.raw.get("scheduling", {})

    @property
    def memory(self) -> dict[str, Any]:
        return self.raw.get("memory", {})

    @property
    def calendar(self) -> dict[str, Any]:
        return self.raw.get("calendar", {})

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def api(self) -> dict[str, Any]:
        return self.raw.get("api", {})

    @property
    def streamlit(self) -> dict[str, Any]:
        return self.raw.get("streamlit", {})


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    """Return cached environment settings."""

    return RuntimeSettings()


@lru_cache(maxsize=1)
def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Load app config from YAML and return as typed wrapper."""

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return AppConfig(raw={})
    data = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(data, dict):
        return AppConfig(raw={})
    return AppConfig(raw=data)
