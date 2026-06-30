"""Configuration models loaded from Hydra YAML and env vars."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model routing and fallback configuration."""

    planner: str = "qwen3:8b"
    researcher: str = "llama3:8b"
    writer: str = "gemma3:12b"
    verifier: str = "deepseek-r1:8b"
    embedder: str = "all-MiniLM-L6-v2"
    fallback_chain: list[str] = Field(
        default_factory=lambda: [
            "qwen3:8b",
            "llama3:8b",
            "mistral:7b",
            "phi4-mini:latest",
        ]
    )


class RetryConfig(BaseModel):
    """Retry policy configuration."""

    max_retries: int = 2
    confidence_threshold: float = 0.72
    backoff_seconds: float = 0.25


class RoutingConfig(BaseModel):
    """Routing heuristics and hard limits."""

    web_search_keywords: list[str] = Field(
        default_factory=lambda: ["latest", "news", "today", "recent", "market"]
    )
    rag_keywords: list[str] = Field(
        default_factory=lambda: ["document", "pdf", "manual", "knowledge", "policy"]
    )
    memory_keywords: list[str] = Field(
        default_factory=lambda: ["previous", "earlier", "before", "history", "remember"]
    )


class MemoryConfig(BaseModel):
    """Persistence backends."""

    sqlite_path: str = "artifacts/langgraph_platform.db"
    chroma_path: str = "artifacts/chroma"
    top_k: int = 5


class MCPConfig(BaseModel):
    """MCP transport and external server config."""

    enabled: bool = True
    transport: str = "stdio"
    http_host: str = "127.0.0.1"
    http_port: int = 9002
    external_servers: list[str] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
    """System and tracing settings."""

    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    enable_gpu_metrics: bool = True
    structured_logs_path: str = "artifacts/logs/platform.jsonl"


class ToolProviderConfig(BaseModel):
    """External API providers and keys."""

    weather_api_key: str | None = None
    currency_api_key: str | None = None
    news_api_key: str | None = None


class AppConfig(BaseModel):
    """Global application configuration."""

    env: str = "dev"
    host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501
    model: ModelConfig = Field(default_factory=ModelConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    tools: ToolProviderConfig = Field(default_factory=ToolProviderConfig)

    @property
    def sqlite_url(self) -> str:
        """SQLAlchemy URL for SQLite backend."""

        return f"sqlite:///{Path(self.memory.sqlite_path)}"


def apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply selected env var overrides on top of file config."""

    env_map: dict[str, tuple[str, str]] = {
        "OLLAMA_PLANNER_MODEL": ("model", "planner"),
        "OLLAMA_RESEARCHER_MODEL": ("model", "researcher"),
        "OLLAMA_WRITER_MODEL": ("model", "writer"),
        "OLLAMA_VERIFIER_MODEL": ("model", "verifier"),
        "PLATFORM_SQLITE_PATH": ("memory", "sqlite_path"),
    }

    for env_var, (section, key) in env_map.items():
        value = __import__("os").environ.get(env_var)
        if value:
            section_obj = getattr(config, section)
            setattr(section_obj, key, value)

    return config


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    """Convert typed config to plain dictionary."""

    return config.model_dump(mode="json")
