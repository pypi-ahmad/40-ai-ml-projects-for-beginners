"""Thin Ollama REST client used by all modules in this project."""

from __future__ import annotations

import logging
import time
from statistics import mean
from typing import Any

import httpx
import psutil

from src.config import get_config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Unified client for completion, chat, embeddings, and benchmark helpers."""

    def __init__(self, base_url: str | None = None, timeout: int | None = None) -> None:
        cfg = get_config()
        self.base_url = (base_url or cfg.ollama_base_url).rstrip("/")
        self.timeout = timeout or cfg.request_timeout_s
        self.max_retries = max(cfg.ollama_max_retries, 0)
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        """Close underlying HTTP connection pool."""

        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Execute HTTP request and normalize error handling."""

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method=method, url=path, json=payload)
                response.raise_for_status()
                return response.json(), None
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                error = f"HTTP {status_code}: {exc.response.text}"
                if status_code >= 500 and attempt < self.max_retries:
                    wait_s = 0.5 * (attempt + 1)
                    logger.warning(
                        "Retrying %s after server error (%s). Attempt %s.",
                        path,
                        status_code,
                        attempt + 1,
                    )
                    time.sleep(wait_s)
                    continue
                logger.error("Ollama HTTP error for %s: %s", path, error)
                return {}, error
            except httpx.RequestError as exc:
                if attempt < self.max_retries:
                    wait_s = 0.5 * (attempt + 1)
                    logger.warning("Retrying %s after connection issue: %s", path, exc)
                    time.sleep(wait_s)
                    continue
                error = f"Connection failed: {exc}. Is Ollama running?"
                logger.error("Ollama request error for %s: %s", path, error)
                return {}, error
            except Exception as exc:  # pragma: no cover - hard to reproduce
                error = str(exc)
                logger.exception("Unexpected Ollama client error on %s", path)
                return {}, error

        return {}, "Request failed after retries."

    def list_models(self) -> list[str]:
        """Return model names currently available in local Ollama instance."""

        payload, error = self._request("GET", "/api/tags")
        if error:
            return []
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
        images: list[str] | None = None,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Call `/api/generate` and return normalized output payload."""

        request_payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            request_payload["system"] = system
        if images:
            request_payload["images"] = images

        started = time.perf_counter()
        response_payload, error = self._request("POST", "/api/generate", request_payload)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return {
            "response": response_payload.get("response", ""),
            "total_duration_ns": response_payload.get("total_duration", 0),
            "eval_count": response_payload.get("eval_count", 0),
            "eval_duration_ns": response_payload.get("eval_duration", 0),
            "latency_ms": round(elapsed_ms, 2),
            "error": error,
        }

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Call `/api/chat` and return normalized output payload."""

        request_payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        started = time.perf_counter()
        response_payload, error = self._request("POST", "/api/chat", request_payload)
        elapsed_ms = (time.perf_counter() - started) * 1000

        message = response_payload.get("message", {})
        return {
            "response": message.get("content", ""),
            "total_duration_ns": response_payload.get("total_duration", 0),
            "eval_count": response_payload.get("eval_count", 0),
            "eval_duration_ns": response_payload.get("eval_duration", 0),
            "latency_ms": round(elapsed_ms, 2),
            "error": error,
        }

    def embed(self, model: str, text: str) -> dict[str, Any]:
        """Generate embedding vector, supporting both embed endpoint variants."""

        for path in ("/api/embed", "/api/embeddings"):
            payload = {"model": model, "input": text}
            response_payload, error = self._request("POST", path, payload)
            if error:
                continue

            embeddings = response_payload.get("embeddings") or [
                response_payload.get("embedding", [])
            ]
            vector = embeddings[0] if embeddings else []
            return {"embedding": vector, "error": None}

        return {"embedding": [], "error": "Embedding request failed."}

    def warmup_model(self, model: str) -> dict[str, Any]:
        """Run tiny prompt to force model load and return readiness status."""

        result = self.generate(
            model=model, prompt="Reply with: ready", max_tokens=8, temperature=0.0
        )
        if result["error"]:
            return {
                "model": model,
                "ready": False,
                "latency_ms": result["latency_ms"],
                "message": result["error"],
            }
        return {
            "model": model,
            "ready": True,
            "latency_ms": result["latency_ms"],
            "message": "Model warmed successfully.",
        }

    def measure_inference_time(
        self,
        model: str,
        prompt: str,
        runs: int = 3,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        """Measure latency, tokens/sec, and memory usage for repeated generation runs."""

        latencies: list[float] = []
        tokens_per_sec_values: list[float] = []
        memory_values: list[float] = []
        errors = 0

        for _ in range(runs):
            process = psutil.Process()
            before_mem_mb = process.memory_info().rss / (1024 * 1024)
            result = self.generate(
                model=model,
                prompt=prompt,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            after_mem_mb = process.memory_info().rss / (1024 * 1024)

            if result["error"]:
                errors += 1
                continue

            latency_ms = float(result["latency_ms"])
            latencies.append(latency_ms)
            memory_values.append(max(after_mem_mb - before_mem_mb, 0.0))

            eval_count = max(int(result.get("eval_count", 0)), 1)
            eval_duration_ns = max(int(result.get("eval_duration_ns", 0)), 1)
            generated_tps = eval_count / (eval_duration_ns / 1_000_000_000)
            tokens_per_sec_values.append(generated_tps)

        if not latencies:
            return {
                "model": model,
                "runs": runs,
                "successful_runs": 0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "mean_tokens_per_sec": 0.0,
                "mean_memory_mb": 0.0,
                "errors": errors,
                "error": "All benchmark runs failed.",
            }

        sorted_latencies = sorted(latencies)
        p95_index = min(int(round(0.95 * (len(sorted_latencies) - 1))), len(sorted_latencies) - 1)

        return {
            "model": model,
            "runs": runs,
            "successful_runs": len(latencies),
            "mean_latency_ms": round(mean(latencies), 2),
            "p95_latency_ms": round(sorted_latencies[p95_index], 2),
            "mean_tokens_per_sec": round(mean(tokens_per_sec_values), 2)
            if tokens_per_sec_values
            else 0.0,
            "mean_memory_mb": round(mean(memory_values), 2) if memory_values else 0.0,
            "latencies_ms": latencies,
            "errors": errors,
            "error": None,
        }
