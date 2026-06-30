"""LLM adapter implementations."""

from __future__ import annotations

import json
from typing import Any

import httpx

from multimodal_ai.adapters.base import LLMAdapter


class OllamaLLMAdapter(LLMAdapter):
    """Ollama text-generation adapter."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        self.name = f"ollama:{model}"
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=120.0)

    def health(self) -> dict[str, Any]:
        try:
            response = self._client.get("/api/ps")
            response.raise_for_status()
            return {"ok": True, "name": self.name}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "name": self.name, "error": str(exc)}

    def complete(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "options": {"temperature": 0.2},
        }
        if system:
            payload["system"] = system

        try:
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()

            text = ""
            done_payload: dict[str, Any] = {}
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                text += chunk.get("response", "")
                done_payload = chunk

            return {
                "text": text.strip(),
                "raw": done_payload,
                "model": self._model,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "text": f"fallback response: ollama backend unavailable ({exc})",
                "raw": {},
                "model": self._model,
            }


class HFTextGenerationAdapter(LLMAdapter):
    """Transformers text-generation adapter with safe fallback."""

    def __init__(self, model_id: str = "Qwen/Qwen3-1.7B") -> None:
        self.name = f"hf:{model_id}"
        self._model_id = model_id
        self._pipeline: Any | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text-generation", model=self._model_id)
        except Exception:  # noqa: BLE001
            self._pipeline = None

    def health(self) -> dict[str, Any]:
        self._load()
        return {"ok": self._pipeline is not None, "name": self.name}

    def complete(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        self._load()
        if self._pipeline is None:
            prefix = f"[{system}] " if system else ""
            return {
                "text": f"{prefix}fallback response: local HF model unavailable",
                "raw": {},
                "model": self._model_id,
            }

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        outputs = self._pipeline(full_prompt, max_new_tokens=256)
        if not outputs:
            return {"text": "", "raw": outputs, "model": self._model_id}
        generated = outputs[0].get("generated_text", "")
        return {
            "text": generated,
            "raw": outputs,
            "model": self._model_id,
        }
