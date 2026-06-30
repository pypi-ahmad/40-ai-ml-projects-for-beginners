"""Application configuration helpers.

Configuration precedence is:
1. ``st.secrets`` (project/cloud secrets)
2. environment variables
3. built-in defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ModelRegistry:
    """Local model tags used by fallback inference providers."""

    sentiment: str = "qwen3.5:2b"
    sentiment_alt: str = "qwen3.5:4b"
    summarization: str = "granite4.1:3b"
    classification: str = "nemotron-3-nano:4b"
    translation: str = "translategemma:4b"


@dataclass(frozen=True)
class ValidationLimits:
    """Input safety limits for end-user text tasks."""

    min_text_length: int = 3
    max_text_length: int = 5000
    max_categories: int = 20
    max_category_length: int = 32


@dataclass(frozen=True)
class AppSettings:
    """Resolved runtime settings for the Streamlit application."""

    hf_api_token: str | None
    ollama_base_url: str
    models: ModelRegistry
    limits: ValidationLimits


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return application settings resolved from secrets/env/defaults."""
    secrets = _load_streamlit_secrets()

    models = ModelRegistry(
        sentiment=_resolve_str("SENTIMENT_MODEL", secrets, ModelRegistry.sentiment),
        sentiment_alt=_resolve_str("SENTIMENT_MODEL_ALT", secrets, ModelRegistry.sentiment_alt),
        summarization=_resolve_str("SUMMARIZATION_MODEL", secrets, ModelRegistry.summarization),
        classification=_resolve_str("CLASSIFICATION_MODEL", secrets, ModelRegistry.classification),
        translation=_resolve_str("TRANSLATION_MODEL", secrets, ModelRegistry.translation),
    )

    limits = ValidationLimits(
        min_text_length=_resolve_int("MIN_TEXT_LENGTH", secrets, ValidationLimits.min_text_length),
        max_text_length=_resolve_int("MAX_TEXT_LENGTH", secrets, ValidationLimits.max_text_length),
        max_categories=_resolve_int("MAX_CATEGORIES", secrets, ValidationLimits.max_categories),
        max_category_length=_resolve_int("MAX_CATEGORY_LENGTH", secrets, ValidationLimits.max_category_length),
    )

    return AppSettings(
        hf_api_token=_resolve_optional_str("HF_API_TOKEN", secrets),
        ollama_base_url=_resolve_str("OLLAMA_BASE_URL", secrets, DEFAULT_OLLAMA_BASE_URL),
        models=models,
        limits=limits,
    )


def is_hf_configured() -> bool:
    """Return True when a Hugging Face token is available."""
    token = get_settings().hf_api_token
    return bool(token and token.strip())


def _load_streamlit_secrets() -> dict[str, Any]:
    """Load Streamlit secrets when available outside strict runtime contexts."""
    try:
        import streamlit as st
    except Exception:
        return {}

    try:
        return {key: st.secrets[key] for key in st.secrets.keys()}
    except Exception:
        return {}


def _resolve_optional_str(key: str, secrets: dict[str, Any]) -> str | None:
    value = _read_preferred(key, secrets)
    if value is None:
        return None

    as_str = str(value).strip()
    return as_str or None


def _resolve_str(key: str, secrets: dict[str, Any], default: str) -> str:
    value = _read_preferred(key, secrets)
    if value is None:
        return default

    as_str = str(value).strip()
    return as_str or default


def _resolve_int(key: str, secrets: dict[str, Any], default: int) -> int:
    value = _read_preferred(key, secrets)
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_preferred(key: str, secrets: dict[str, Any]) -> Any:
    if key in secrets:
        return secrets[key]

    model_section = secrets.get("models")
    if isinstance(model_section, dict):
        nested_key = key.lower()
        if nested_key in model_section:
            return model_section[nested_key]

    return os.getenv(key)
