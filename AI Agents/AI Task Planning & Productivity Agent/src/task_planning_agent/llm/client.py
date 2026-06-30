"""LLM client layer using LangChain Ollama with fallback support."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


class LocalLLMClient:
    """Wrapper around ChatOllama with model fallback chain."""

    def __init__(self, base_url: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, candidate_models: Iterable[str]) -> tuple[str, str]:
        last_error: Exception | None = None
        for model_name in candidate_models:
            try:
                model = ChatOllama(model=model_name, base_url=self.base_url, timeout=self.timeout_seconds)
                response = model.invoke(prompt)
                content = getattr(response, "content", str(response))
                logger.info("model_selected=%s", model_name)
                return str(content), model_name
            except Exception as exc:  # pragma: no cover - runtime integration path
                last_error = exc
                logger.warning("model_failed=%s error=%s", model_name, exc)
        if last_error is None:
            raise RuntimeError("No candidate models configured")
        raise RuntimeError(f"All fallback models failed: {last_error}")
