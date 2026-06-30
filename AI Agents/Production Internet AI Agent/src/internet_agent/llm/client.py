"""LangChain-Ollama wrappers for model inference."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from internet_agent.config import Settings


class OllamaClient:
    """Multi-model Ollama client with task-specific model routing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._availability_checked_at = 0.0
        self._ollama_available = True

    def _model(self, model_name: str, temperature: float | None = None) -> ChatOllama:
        return ChatOllama(
            model=model_name,
            base_url=self.settings.llm.base_url,
            temperature=self.settings.llm.temperature if temperature is None else temperature,
            num_predict=self.settings.llm.max_tokens,
            timeout=self.settings.llm.request_timeout_seconds,
        )

    async def _is_ollama_available(self) -> bool:
        now = time.time()
        if now - self._availability_checked_at < 5:
            return self._ollama_available

        self._availability_checked_at = now
        health_url = self.settings.llm.base_url.rstrip("/") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(health_url)
            self._ollama_available = response.status_code < 500
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    async def ask(
        self,
        task_model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        if not await self._is_ollama_available():
            return "Model inference unavailable. Check Ollama service/model setup."
        try:
            model = self._model(task_model, temperature=temperature)
            response = await model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return str(response.content).strip()
        except Exception as exc:  # noqa: BLE001
            return (
                "Model inference unavailable. "
                f"Check Ollama service/model setup. Error: {exc}"
            )

    async def ask_json(
        self,
        task_model: str,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            f"{user_prompt}\n\n"
            "Return strict JSON object only. Do not add markdown fences or extra text."
        )
        text = await self.ask(task_model=task_model, system_prompt=system_prompt, user_prompt=prompt)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return fallback
