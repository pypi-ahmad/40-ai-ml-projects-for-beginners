import json
import time
from typing import cast

import httpx


class OllamaClient:
    BASE = "http://localhost:11434"

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=self.BASE, timeout=120)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        raw: bool = False,
        temperature: float = 0.1,
        images: list[str] | None = None,
    ) -> dict:
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "raw": raw,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images
        r = self._client.post("/api/generate", json=payload)
        r.raise_for_status()
        return self._stream_aggregate(r)

    def chat(self, model: str, messages: list[dict], temperature: float = 0.7) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        r = self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        return self._stream_aggregate(r)

    def embed(self, model: str, input_text: str) -> list[float]:
        payload: dict = {"model": model, "input": input_text}
        r = self._client.post("/api/embed", json=payload)
        r.raise_for_status()
        data = r.json()
        return cast(list[float], data.get("embeddings", [[]])[0])

    def measure_inference_time(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        raw: bool = False,
        temperature: float = 0.1,
    ) -> dict:
        start = time.perf_counter()
        result = self.generate(model, prompt, system, raw, temperature)
        elapsed = time.perf_counter() - start
        return {"latency_s": elapsed, "tokens": result.get("eval_count", 0), "full_result": result}

    @staticmethod
    def _stream_aggregate(response: httpx.Response) -> dict:
        aggregated: dict = {}
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "response" in chunk and "response" in aggregated:
                aggregated["response"] += chunk["response"]
                chunk.pop("response")
            aggregated.update(chunk)
        return aggregated
