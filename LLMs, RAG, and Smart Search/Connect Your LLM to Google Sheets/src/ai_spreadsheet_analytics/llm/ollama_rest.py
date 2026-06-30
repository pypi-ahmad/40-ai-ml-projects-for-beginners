"""Direct Ollama REST API client."""

from __future__ import annotations

import time

import httpx

from ai_spreadsheet_analytics.exceptions import LLMError
from ai_spreadsheet_analytics.llm.base import LLMResponse


class OllamaRESTClient:
    """Direct REST client for Ollama inference."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_sec: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    async def agenerate(self, model: str, prompt: str, system: str = "", temperature: float = 0.0) -> LLMResponse:
        """Generate response via /api/generate.

        Args:
            model: Ollama model tag.
            prompt: User prompt.
            system: Optional system instructions.
            temperature: Decoding temperature, fixed at zero for analytics.

        Returns:
            Standardized LLM response.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code >= 400:
            raise LLMError(f"Ollama API error {response.status_code}: {response.text}")

        raw = response.json()
        text = raw.get("response", "").strip()
        tokens = max(len(text.split()), 1)
        return LLMResponse(text=text, latency_ms=elapsed_ms, token_estimate=tokens, raw=raw)

    async def ahealth(self) -> bool:
        """Check Ollama availability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False
