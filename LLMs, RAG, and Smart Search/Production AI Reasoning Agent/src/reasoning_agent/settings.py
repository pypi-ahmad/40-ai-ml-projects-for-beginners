"""Application settings and config loading."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings from environment + YAML."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"

    primary_model: str = "qwen3:8b"
    compare_models: str = "llama3.1:8b,granite4.1:3b,deepseek-r1"
    embedding_model: str = "qwen3-embedding:4b"
    temperature: float = 0.1
    max_tokens: int = 1200

    max_iterations: int = 8
    max_retries: int = 2
    request_timeout_seconds: int = 30

    weather_provider: str = "open_meteo"
    weather_api_key: str = ""
    currency_provider: str = "frankfurter"
    currency_api_key: str = ""
    news_provider: str = "gnews"
    news_api_key: str = ""

    chroma_dir: str = "artifacts/chroma"
    memory_retention_days: int = 30
    memory_top_k: int = 5

    log_dir: str = "logs"
    log_level: str = "INFO"
    log_json: bool = True

    trace_visibility: str = "safe_summary"
    benchmark_dataset_path: str = "benchmarks/prompts.jsonl"
    benchmark_output_dir: str = "artifacts/reports"

    enabled_tools: list[str] = Field(default_factory=list)

    @property
    def compare_model_list(self) -> list[str]:
        """List form for compare models."""

        return [m.strip() for m in self.compare_models.split(",") if m.strip()]


@lru_cache(maxsize=1)
def load_settings(config_path: str | Path = "configs/config.yaml") -> Settings:
    """Load settings from YAML then apply env overrides.

    Args:
        config_path: YAML config file path.

    Returns:
        Fully resolved Settings instance.
    """

    path = Path(config_path)
    raw: dict[str, Any] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        model = loaded.get("models", {})
        graph = loaded.get("graph", {})
        memory = loaded.get("memory", {})
        tools = loaded.get("tools", {})
        logging_cfg = loaded.get("logging", {})
        benchmark = loaded.get("benchmark", {})
        app = loaded.get("app", {})

        raw = {
            "primary_model": model.get("primary", "qwen3:8b"),
            "compare_models": ",".join(model.get("compare", [])),
            "embedding_model": model.get("embedding", "qwen3-embedding:4b"),
            "temperature": model.get("temperature", 0.1),
            "max_tokens": model.get("max_tokens", 1200),
            "max_iterations": graph.get("max_iterations", 8),
            "max_retries": graph.get("max_retries", 2),
            "request_timeout_seconds": graph.get("timeout_seconds", 30),
            "chroma_dir": memory.get("chroma_dir", "artifacts/chroma"),
            "memory_retention_days": memory.get("retention_days", 30),
            "memory_top_k": memory.get("top_k", 5),
            "log_dir": logging_cfg.get("dir", "logs"),
            "log_level": logging_cfg.get("level", "INFO"),
            "log_json": logging_cfg.get("json", True),
            "trace_visibility": app.get("trace_visibility", "safe_summary"),
            "benchmark_dataset_path": benchmark.get("dataset_path", "benchmarks/prompts.jsonl"),
            "benchmark_output_dir": benchmark.get("output_dir", "artifacts/reports"),
            "enabled_tools": tools.get("enabled", []),
        }

    # Enforce precedence contract: env > yaml > defaults.
    env_map = {
        "ollama_base_url": "OLLAMA_BASE_URL",
        "primary_model": "PRIMARY_MODEL",
        "compare_models": "COMPARE_MODELS",
        "embedding_model": "EMBEDDING_MODEL",
        "temperature": "TEMPERATURE",
        "max_tokens": "MAX_TOKENS",
        "max_iterations": "MAX_ITERATIONS",
        "max_retries": "MAX_RETRIES",
        "request_timeout_seconds": "REQUEST_TIMEOUT_SECONDS",
        "weather_provider": "WEATHER_PROVIDER",
        "weather_api_key": "WEATHER_API_KEY",
        "currency_provider": "CURRENCY_PROVIDER",
        "currency_api_key": "CURRENCY_API_KEY",
        "news_provider": "NEWS_PROVIDER",
        "news_api_key": "NEWS_API_KEY",
        "chroma_dir": "CHROMA_DIR",
        "memory_retention_days": "MEMORY_RETENTION_DAYS",
        "memory_top_k": "MEMORY_TOP_K",
        "log_dir": "LOG_DIR",
        "log_level": "LOG_LEVEL",
        "log_json": "LOG_JSON",
        "trace_visibility": "TRACE_VISIBILITY",
        "benchmark_dataset_path": "BENCHMARK_DATASET_PATH",
        "benchmark_output_dir": "BENCHMARK_OUTPUT_DIR",
    }
    for field_name, env_var in env_map.items():
        if env_var in os.environ and field_name in raw:
            raw.pop(field_name, None)

    return Settings(**raw)
