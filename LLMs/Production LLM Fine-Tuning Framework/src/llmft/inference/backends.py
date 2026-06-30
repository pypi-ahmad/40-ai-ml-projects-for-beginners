"""Inference router for Transformers, vLLM, and Ollama backends."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from llmft.config.schemas import InferenceConfig
from llmft.utils.logging import get_logger


class InferenceBackend(Protocol):
    """Protocol for async generation backends."""

    async def generate(self, prompt: str) -> str:
        """Generate response for one prompt."""


@dataclass(slots=True)
class LatencyRecord:
    """Latency benchmark result record."""

    backend: str
    prompt_count: int
    mean_latency_ms: float


class InferenceRouter:
    """Route inference requests to configured backend."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = get_logger("llmft.inference")
        self._backend = self._select_backend(config.backend)

    async def generate(self, prompt: str) -> str:
        """Generate one response."""
        return await self._backend.generate(prompt)

    async def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a prompt batch."""
        tasks = [self.generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def benchmark(self, prompts: list[str]) -> LatencyRecord:
        """Measure average latency for selected backend."""
        start = time.perf_counter()
        await self.generate_batch(prompts)
        elapsed = (time.perf_counter() - start) * 1000
        mean_ms = elapsed / max(1, len(prompts))
        return LatencyRecord(
            backend=self.config.backend,
            prompt_count=len(prompts),
            mean_latency_ms=round(mean_ms, 3),
        )

    def _select_backend(self, backend_name: str) -> InferenceBackend:
        name = backend_name.lower()
        if name == "transformers":
            return TransformersBackend(self.config)
        if name == "vllm":
            return VLLMBackend(self.config)
        if name == "ollama":
            return OllamaBackend(self.config)
        raise ValueError(f"Unsupported backend: {backend_name}")


class TransformersBackend:
    """Transformers local backend."""

    _pipeline_cache: dict[str, Any] = {}

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.logger = get_logger("llmft.inference.transformers")
        self.model_id = os.getenv("LLMFT_TRANSFORMERS_MODEL", "sshleifer/tiny-gpt2")

    def _get_pipeline(self):
        if self.model_id in self._pipeline_cache:
            return self._pipeline_cache[self.model_id]
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("transformers package missing") from exc

        pipe = pipeline("text-generation", model=self.model_id)
        self._pipeline_cache[self.model_id] = pipe
        self.logger.info("loaded inference model: %s", self.model_id)
        return pipe

    async def generate(self, prompt: str) -> str:
        pipe = self._get_pipeline()

        def _run() -> str:
            result = pipe(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return str(result[0].get("generated_text", ""))

        return await asyncio.to_thread(_run)


class VLLMBackend:
    """vLLM OpenAI-compatible backend."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    async def generate(self, prompt: str) -> str:
        if not self.config.enable_remote_backends:
            raise RuntimeError("vLLM backend disabled in config")

        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        url = f"{self.config.vllm_host.rstrip('/')}/v1/completions"

        def _request() -> str:
            response = _http_post_json(url, payload, timeout=self.config.request_timeout_seconds)
            choices = response.get("choices", [])
            if not choices:
                raise RuntimeError("vLLM response missing choices")
            return str(choices[0].get("text", ""))

        return await asyncio.to_thread(_request)


class OllamaBackend:
    """Ollama generate API backend."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    async def generate(self, prompt: str) -> str:
        if not self.config.enable_remote_backends:
            raise RuntimeError("Ollama backend disabled in config")

        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            },
        }
        url = f"{self.config.ollama_host.rstrip('/')}/api/generate"

        def _request() -> str:
            response = _http_post_json(url, payload, timeout=self.config.request_timeout_seconds)
            if "response" not in response:
                raise RuntimeError("Ollama response missing text")
            return str(response.get("response", ""))

        return await asyncio.to_thread(_request)


def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            content = response.read().decode("utf-8")
            return json.loads(content)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {text[:200]}") from exc
