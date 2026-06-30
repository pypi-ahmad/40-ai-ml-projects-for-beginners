"""Configuration models and loader."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from reasoning_agent.constants import DEFAULT_CONFIG_PATH


class LLMConfig(BaseModel):
    """Configuration for model provider and defaults."""

    base_url: str = "http://127.0.0.1:11434"
    primary_model: str = "qwen3:8b"
    benchmark_models: list[str] = Field(
        default_factory=lambda: ["qwen3:8b", "llama3.1:8b", "granite4.1:3b", "deepseek-r1"]
    )
    llm_judge_model: str = "granite4.1:3b"
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout_seconds: int = 120
    auto_pull_missing_models: bool = True


class RetryConfig(BaseModel):
    """Retry budget configuration."""

    max_retries: int = 2
    backoff_seconds: float = 1.5


class MemoryConfig(BaseModel):
    """Memory stack configuration."""

    chroma_enabled: bool = True
    chroma_path: str = "artifacts/chroma"
    memory_top_k: int = 5
    conversation_window: int = 20


class ToolConfig(BaseModel):
    """Tool system configuration."""

    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
    optional_tools: list[str] = Field(default_factory=list)
    enable_python_tool: bool = False
    python_timeout_seconds: int = 5
    python_memory_limit_mb: int = 128
    workspace_root: str = "."


class AgentConfig(BaseModel):
    """Agent orchestration configuration."""

    max_iterations: int = 10
    graph_timeout_seconds: int = 30
    runtime_mode: Literal["graph", "fallback", "auto"] = "graph"
    graph_fallback_on_error: bool = True
    reasoning_mode: Literal["react", "reflect", "tot"] = "react"
    use_llm_for_planning: bool = True
    use_llm_for_response: bool = True


class LoggingConfig(BaseModel):
    """Logging and observability configuration."""

    level: str = "INFO"
    json_logs: bool = True
    run_log_path: str = "logs/agent_runs.jsonl"


class UIConfig(BaseModel):
    """Streamlit configuration."""

    title: str = "Production AI Reasoning Agent"
    dark_mode_default: bool = True


class Settings(BaseModel):
    """Merged environment and YAML settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must be mapping: {path}")
        return data


def _parse_env_overrides(prefix: str = "AGENT__") -> dict:
    overrides: dict = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key.removeprefix(prefix).lower().split("__")
        cursor = overrides
        for segment in path[:-1]:
            if segment not in cursor or not isinstance(cursor[segment], dict):
                cursor[segment] = {}
            cursor = cursor[segment]
        cursor[path[-1]] = yaml.safe_load(value)
    return overrides


def _env_signature(prefix: str = "AGENT__") -> tuple[tuple[str, str], ...]:
    """Return deterministic cache key for relevant environment variables."""

    return tuple(sorted((key, value) for key, value in os.environ.items() if key.startswith(prefix)))


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_deprecated_keys(raw: dict) -> dict:
    """Map old config keys to current schema without breaking existing YAML files."""

    normalized = dict(raw)
    agent_cfg = normalized.get("agent")
    if isinstance(agent_cfg, dict):
        agent_cfg = dict(agent_cfg)
        use_langgraph_runtime = agent_cfg.pop("use_langgraph_runtime", None)
        if "runtime_mode" not in agent_cfg and isinstance(use_langgraph_runtime, bool):
            agent_cfg["runtime_mode"] = "graph" if use_langgraph_runtime else "fallback"
        agent_cfg.pop("stream_tokens", None)
        normalized["agent"] = agent_cfg
    return normalized


@lru_cache(maxsize=8)
def _get_settings_cached(path: str, env_signature: tuple[tuple[str, str], ...]) -> Settings:
    """Cached settings loader keyed by config path + relevant environment."""

    del env_signature  # only used as cache key
    yaml_data = _load_yaml(Path(path))
    env_data = _parse_env_overrides(prefix="AGENT__")
    merged = _deep_merge(yaml_data, env_data)
    normalized = _normalize_deprecated_keys(merged)
    return Settings.model_validate(normalized)


def get_settings(config_path: str | Path | None = None, refresh: bool = False) -> Settings:
    """Load settings with environment overrides.

    Args:
        config_path: Optional path to YAML file.
        refresh: Clear cache before loading.

    Returns:
        Parsed settings object.
    """

    if refresh:
        clear_settings_cache()
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return _get_settings_cached(str(path), _env_signature())


def clear_settings_cache() -> None:
    """Clear cached settings to pick up environment updates."""

    _get_settings_cached.cache_clear()
