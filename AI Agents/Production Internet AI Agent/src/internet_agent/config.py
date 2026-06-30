"""Configuration loading and validation."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Production Internet AI Agent"
    environment: str = "dev"


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    planning_model: str = "qwen3:8b"
    reasoning_model: str = "llama3.1:8b"
    summarization_model: str = "gemma3:4b"
    verification_model: str = "deepseek-r1:8b"
    reflection_model: str = "mistral:7b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.2
    max_tokens: int = 1200
    request_timeout_seconds: int = 120


class ModelSupportConfig(BaseModel):
    supported_families: list[str] = Field(
        default_factory=lambda: ["llama3", "qwen3", "gemma3", "deepseek", "phi4-mini", "mistral"]
    )


class AgentConfig(BaseModel):
    max_iterations: int = 6
    max_verification_loops: int = 2
    verification_confidence_threshold: float = 0.72
    top_k_sources: int = 5
    checkpointer_enabled: bool = False
    graph_timeout_seconds: int = 45


class SearchConfig(BaseModel):
    providers: list[str] = Field(
        default_factory=lambda: ["duckduckgo", "wikipedia", "github", "news"]
    )
    default_max_results: int = 5
    request_timeout_seconds: int = 20


class RetrievalConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 120
    retrieval_top_k: int = 6
    max_urls_per_query: int = 8
    max_content_chars: int = 24_000


class MemoryConfig(BaseModel):
    sqlite_url: str = "sqlite:///./artifacts/memory.db"
    chroma_path: str = "artifacts/chroma"
    chroma_collection: str = "internet_docs"
    conversation_window: int = 30
    memory_top_k: int = 5


class CacheConfig(BaseModel):
    enabled: bool = True
    default_ttl_seconds: int = 1800
    embedding_ttl_seconds: int = 259_200
    report_ttl_seconds: int = 86_400
    redis_url: str = ""


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    require_api_key: bool = False
    api_key_env: str = "INTERNET_AGENT_API_KEY"


class StreamlitConfig(BaseModel):
    title: str = "Production Internet AI Agent"
    dark_mode_default: bool = True


class ReportsConfig(BaseModel):
    output_dir: str = "outputs/reports"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = True
    log_path: str = "artifacts/logs/agent.jsonl"


class MonitoringConfig(BaseModel):
    metrics_enabled: bool = True
    mlflow_enabled: bool = True
    mlflow_experiment: str = "internet_agent"
    mlflow_tracking_uri: str = ""


class PluginConfig(BaseModel):
    tool_factories: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    models: ModelSupportConfig = Field(default_factory=ModelSupportConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    reports: ReportsConfig = Field(default_factory=ReportsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(prefix: str = "INTERNET_AGENT__") -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key.removeprefix(prefix).lower().split("__")
        cursor = overrides
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[path[-1]] = OmegaConf.create({"x": value}).x
    return overrides


@lru_cache(maxsize=4)
def get_settings(config_path: str | Path = "configs/config.yaml") -> Settings:
    """Load settings from YAML + environment overrides."""

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return Settings()

    raw = _load_config_mapping(cfg_path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {cfg_path} must be a mapping")

    merged = _deep_merge(raw, _env_overrides())
    settings = Settings.model_validate(merged)
    ensure_runtime_dirs(settings)
    return settings


def ensure_runtime_dirs(settings: Settings) -> None:
    """Create runtime directories expected by the platform."""

    dirs = {
        Path(settings.memory.chroma_path),
        Path(settings.reports.output_dir),
        Path(settings.logging.log_path).parent,
        Path("outputs/screenshots"),
        Path("outputs/analytics"),
        Path("artifacts"),
    }
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def _load_config_mapping(cfg_path: Path) -> dict[str, Any]:
    """Load config through Hydra compose with YAML fallback."""

    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent.resolve())):
            cfg = compose(config_name=cfg_path.stem)
        raw = OmegaConf.to_container(cfg, resolve=True)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        raw = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        return raw if isinstance(raw, dict) else {}
