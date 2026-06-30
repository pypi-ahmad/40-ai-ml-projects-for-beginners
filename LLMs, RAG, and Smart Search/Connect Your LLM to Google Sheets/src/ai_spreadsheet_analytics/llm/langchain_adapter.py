"""LangChain-based Ollama adapter."""

from __future__ import annotations

import time

from ai_spreadsheet_analytics.exceptions import LLMError
from ai_spreadsheet_analytics.llm.base import LLMResponse

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
except Exception:  # noqa: BLE001
    ChatOllama = None
    HumanMessage = None
    SystemMessage = None


class LangChainOllamaAdapter:
    """LangChain orchestration path for Ollama."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url

    async def agenerate(self, model: str, prompt: str, system: str = "", temperature: float = 0.0) -> LLMResponse:
        """Generate response through LangChain ChatOllama."""
        if ChatOllama is None or HumanMessage is None or SystemMessage is None:
            raise LLMError("LangChain Ollama dependencies not available")

        llm = ChatOllama(model=model, base_url=self.base_url, temperature=temperature)
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)] if system else [HumanMessage(content=prompt)]
        start = time.perf_counter()
        response = await llm.ainvoke(messages)
        elapsed_ms = (time.perf_counter() - start) * 1000

        content = response.content if isinstance(response.content, str) else str(response.content)
        return LLMResponse(
            text=content.strip(),
            latency_ms=elapsed_ms,
            token_estimate=max(len(content.split()), 1),
            raw={"adapter": "langchain"},
        )
