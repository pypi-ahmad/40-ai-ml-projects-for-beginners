"""Thin Ollama client for generation and structured fixup."""

from __future__ import annotations

import json

import httpx

from resume_ai.config.loader import AppConfig


class OllamaLLM:
    """HTTP client for local Ollama API."""

    def __init__(self, config: AppConfig):
        self.config = config

    def generate(self, prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
        timeout_s = float(min(self.config.retries.timeout_seconds, 5))
        payload = {
            "model": model or self.config.models.reasoning_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=1.0)) as client:
            try:
                response = client.post(f"{self.config.models.ollama_base_url}/api/generate", json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception:
                return ""

    def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        retries: int | None = None,
    ) -> dict:
        max_retries = retries if retries is not None else self.config.retries.llm_retries
        attempts = max_retries + 1
        for _ in range(attempts):
            text = self.generate(prompt=prompt, model=model, temperature=0.0)
            if not text.strip():
                continue
            candidate = text.strip().strip("`")
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {}
