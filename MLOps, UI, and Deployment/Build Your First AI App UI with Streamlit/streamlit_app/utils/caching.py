"""Caching wrappers for model calls.

`st.cache_resource` is used for long-lived resources.
`st.cache_data` is used for deterministic task outputs.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from streamlit_app.config import APP_CONFIG
from streamlit_app.utils import models


@st.cache_resource(show_spinner="Initializing local model client...")
def get_ollama_client() -> dict[str, Any]:
    """Return immutable client metadata for app diagnostics.

    We keep this as small structure to demonstrate `st.cache_resource` semantics
    (initialized once per process).
    """
    return {
        "provider": "ollama",
        "host": APP_CONFIG.ollama_host,
        "timeout_seconds": APP_CONFIG.request_timeout_seconds,
    }


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_sentiment(text: str, model: str) -> dict[str, Any]:
    get_ollama_client()
    return models.analyze_sentiment(text=text, model=model)


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_summarize(text: str, max_length: int, model: str, temperature: float) -> str:
    get_ollama_client()
    return models.summarize_text(text=text, max_length=max_length, model=model, temperature=temperature)


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_classify(text: str, categories: tuple[str, ...], model: str) -> dict[str, Any]:
    get_ollama_client()
    return models.classify_text(text=text, categories=list(categories), model=model)


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_translate(text: str, target_lang: str, model: str) -> str:
    get_ollama_client()
    return models.translate_text(text=text, target_lang=target_lang, model=model)


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_chat(
    messages: tuple[tuple[str, str], ...],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    get_ollama_client()
    payload = [{"role": role, "content": content} for role, content in messages]
    return models.chat_response(payload, model=model, temperature=temperature, max_tokens=max_tokens)


@st.cache_data(show_spinner=False, ttl=APP_CONFIG.cache_ttl_seconds)
def cached_ocr(content: bytes, filename: str, primary_model: str, fallback_model: str) -> str:
    get_ollama_client()
    return models.ocr_analyze(
        content=content,
        filename=filename,
        primary_model=primary_model,
        fallback_model=fallback_model,
    )


def benchmark_with_cache(
    model: str,
    prompt: str,
    runs: int,
    system_prompt: str | None,
    temperature: float,
) -> dict[str, Any]:
    """Run benchmark through model module (no cache by design).

    Benchmark runs should remain uncached to reflect true latency distribution.
    """
    return models.benchmark_inference(
        model=model,
        prompt=prompt,
        system=system_prompt,
        runs=runs,
        temperature=temperature,
    )


def compare_cached_vs_uncached_sentiment(text: str, model: str) -> dict[str, float]:
    """Measure response-time difference between uncached and cached calls."""
    start_uncached = time.perf_counter()
    _ = models.analyze_sentiment(text=text, model=model)
    uncached_elapsed = time.perf_counter() - start_uncached

    start_cached_first = time.perf_counter()
    _ = cached_sentiment(text=text, model=model)
    cached_first_elapsed = time.perf_counter() - start_cached_first

    start_cached_second = time.perf_counter()
    _ = cached_sentiment(text=text, model=model)
    cached_second_elapsed = time.perf_counter() - start_cached_second

    return {
        "uncached_seconds": uncached_elapsed,
        "cached_first_seconds": cached_first_elapsed,
        "cached_second_seconds": cached_second_elapsed,
    }
