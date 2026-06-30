"""Central configuration for the Streamlit AI Application Studio.

This module keeps runtime knobs in one place so the app, tests, scripts,
and notebooks use the same defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(slots=True)
class ModelConfig:
    """Model identifiers used across mini-app pages."""

    sentiment: str = "granite4.1:3b"
    summarization: str = "qwen3.5:4b"
    classification: str = "granite4.1:3b"
    translation: str = "translategemma:4b"
    chat: str = "qwen3.5:4b"
    chat_fast: str = "qwen3.5:2b"
    embedding: str = "qwen3-embedding:4b"
    ocr_primary: str = "glm-ocr:latest"
    ocr_fallback: str = "deepseek-ocr:latest"
    benchmark_extra: str = "nemotron-3-nano:4b"

    @property
    def benchmark_models(self) -> list[str]:
        """Return model list used in benchmark page and notebook."""
        return [
            self.chat_fast,
            self.chat,
            self.sentiment,
            self.benchmark_extra,
        ]


@dataclass(slots=True)
class AppConfig:
    """High-level app settings with environment variable overrides."""

    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
    )
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "1800")))
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.2")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "512")))
    benchmark_runs: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_RUNS", "3")))
    models: ModelConfig = field(default_factory=ModelConfig)


APP_CONFIG = AppConfig()
