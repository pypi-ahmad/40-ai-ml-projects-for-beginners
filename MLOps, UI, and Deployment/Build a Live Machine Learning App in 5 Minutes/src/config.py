"""Centralized runtime configuration for local Gradio + Ollama app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration resolved from environment variables."""

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    request_timeout_s: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama_max_retries: int = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))

    sentiment_model: str = os.getenv("MODEL_SENTIMENT", "qwen3.5:2b")
    summarization_model: str = os.getenv("MODEL_SUMMARY", "granite4.1:3b")
    translation_model: str = os.getenv("MODEL_TRANSLATION", "translategemma:4b")
    chat_model: str = os.getenv("MODEL_CHAT", "qwen3.5:4b")
    ocr_primary_model: str = os.getenv("MODEL_OCR_PRIMARY", "glm-ocr:latest")
    ocr_fallback_model: str = os.getenv("MODEL_OCR_FALLBACK", "deepseek-ocr:latest")
    embedding_model: str = os.getenv("MODEL_EMBEDDING", "qwen3-embedding:4b")

    document_max_pages: int = int(os.getenv("DOCUMENT_MAX_PAGES", "5"))
    document_max_file_size_mb: int = int(os.getenv("DOCUMENT_MAX_FILE_SIZE_MB", "20"))
    document_max_image_pixels: int = int(os.getenv("DOCUMENT_MAX_IMAGE_PIXELS", "24000000"))
    chat_max_turns: int = int(os.getenv("CHAT_MAX_TURNS", "20"))

    benchmark_runs: int = int(os.getenv("BENCHMARK_RUNS", "3"))
    benchmark_fast_runs: int = int(os.getenv("BENCHMARK_FAST_RUNS", "1"))
    benchmark_full_runs: int = int(os.getenv("BENCHMARK_FULL_RUNS", "3"))


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached immutable app configuration."""

    return AppConfig()
