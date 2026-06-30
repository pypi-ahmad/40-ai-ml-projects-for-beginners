from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "production-mcp-server"
    environment: str = "dev"


class ModelsConfig(BaseModel):
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "qwen3:8b"
    supported_models: list[str] = Field(default_factory=list)


class TransportConfig(BaseModel):
    runtime: Literal["fastmcp", "mcp"] = "fastmcp"
    mode: Literal["stdio", "sse", "http", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8001
    sse_path: str = "/events"
    http_path: str = "/mcp"


class AuthConfig(BaseModel):
    enabled: bool = True
    read_only_mode: bool = False
    api_keys: dict[str, str] = Field(default_factory=dict)


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: int = 900


class MemoryConfig(BaseModel):
    sqlite_path: str = "database/memory.db"
    chroma_path: str = "chroma_db"
    chroma_enabled: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5


class PluginConfig(BaseModel):
    enabled: bool = True
    directory: str = "plugins"
    allowlist_manifest: str = "configs/plugins_allowlist.yaml"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True
    file_path: str = "logs/server.log"


class MonitoringConfig(BaseModel):
    enabled: bool = True
    sample_interval_seconds: int = 5
    mlflow_tracking_uri: str = "sqlite:///database/mlflow.db"


class SchedulerConfig(BaseModel):
    enabled: bool = True
    timezone: str = "UTC"
    index_every_minutes: int = 30
    cleanup_every_minutes: int = 60
    report_every_minutes: int = 120


class ShellConfig(BaseModel):
    whitelist: list[str] = Field(default_factory=lambda: ["ls", "pwd", "cat", "head", "tail", "wc"])


class ExternalConfig(BaseModel):
    news_api_key_env: str = "NEWS_API_KEY"
    github_token_env: str = "GITHUB_TOKEN"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    shell: ShellConfig = Field(default_factory=ShellConfig)
    external: ExternalConfig = Field(default_factory=ExternalConfig)


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _set_nested(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = mapping
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key, value in os.environ.items():
        if not key.startswith("MCP_SERVER__"):
            continue
        parts = key.removeprefix("MCP_SERVER__").lower().split("__")
        _set_nested(result, parts, _coerce_scalar(value))
    return result


def load_settings(path: str | Path | None = None) -> Settings:
    path_obj = Path(path) if path else Path("configs/default.yaml")
    if not path_obj.exists():
        return Settings()

    with path_obj.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    payload = _apply_env_overrides(payload)
    settings = Settings.model_validate(payload)

    base_dir = path_obj.parent
    settings.memory.sqlite_path = str((base_dir.parent / settings.memory.sqlite_path).resolve())
    settings.memory.chroma_path = str((base_dir.parent / settings.memory.chroma_path).resolve())
    settings.plugins.directory = str((base_dir.parent / settings.plugins.directory).resolve())
    settings.plugins.allowlist_manifest = str(
        (base_dir.parent / settings.plugins.allowlist_manifest).resolve()
    )
    settings.logging.file_path = str((base_dir.parent / settings.logging.file_path).resolve())
    return settings
