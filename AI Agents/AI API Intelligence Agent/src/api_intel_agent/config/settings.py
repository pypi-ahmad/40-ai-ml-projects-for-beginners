"""Configuration loader using YAML + environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen3:8b"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 30
    supported_models: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    max_iterations: int = 6
    max_parallel_calls: int = 4
    retry_max_attempts: int = 3
    retry_backoff_seconds: float = 1.5
    default_output_format: str = "markdown"


class AuthConfig(BaseModel):
    jwt_secret_env: str = "AGENT_JWT_SECRET"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_minutes: int = 1440


class CacheConfig(BaseModel):
    enabled: bool = True
    backend: str = "sqlite"
    sqlite_path: str = "artifacts/cache/cache.db"
    redis_url: str = ""
    default_ttl_seconds: int = 900


class MemoryConfig(BaseModel):
    sqlite_path: str = "artifacts/memory/agent_memory.db"
    chroma_enabled: bool = True
    chroma_path: str = "artifacts/memory/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5


class MonitoringConfig(BaseModel):
    metrics_enabled: bool = True
    collect_gpu: bool = True
    export_prometheus: bool = False


class SchedulerConfig(BaseModel):
    enabled: bool = True
    report_cron: str = "0 */6 * * *"


class ReportsConfig(BaseModel):
    output_dir: str = "artifacts/reports"
    include_charts: bool = True
    include_sources: bool = True


class UIConfig(BaseModel):
    streamlit_title: str = "AI API Intelligence Dashboard"
    dark_mode_default: bool = True


class AppConfigModel(BaseModel):
    app: dict[str, Any] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    apis: dict[str, dict[str, Any]] = Field(default_factory=dict)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)


class EnvSettings(BaseSettings):
    """Environment variables that must never be committed."""

    agent_jwt_secret: str = "change-me-32-char-minimum-secret-key"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    prefix = "AGENT__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


@lru_cache(maxsize=1)
def load_settings(path: str | Path = "configs/settings.yaml") -> AppConfigModel:
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
    data = _apply_env_overrides(data)
    model = AppConfigModel.model_validate(data)

    # Ensure output directories exist for runtime components.
    Path(model.cache.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model.memory.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model.memory.chroma_path).mkdir(parents=True, exist_ok=True)
    Path(model.reports.output_dir).mkdir(parents=True, exist_ok=True)
    return model


def get_secret(settings: AppConfigModel, env: EnvSettings | None = None) -> str:
    env = env or EnvSettings()
    secret_name = settings.auth.jwt_secret_env.lower()
    if secret_name == "agent_jwt_secret":
        return env.agent_jwt_secret
    return os.getenv(settings.auth.jwt_secret_env, "change-me")
