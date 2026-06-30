"""Local multi-turn chat engine with bounded conversation memory."""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import ChatResult, ChatTurn, ErrorInfo

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are local AI assistant running via Ollama.
Be accurate, concise, and transparent when uncertain."""


class ChatEngine:
    """Maintain chat history and generate replies with context trimming."""

    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: OllamaClient | None = None,
    ) -> None:
        cfg = get_config()
        self.model = model or cfg.chat_model
        self.max_turns = max_turns or cfg.chat_max_turns
        self.system_prompt = system_prompt
        self.client = client or OllamaClient()
        self._owns_client = client is None
        self._history: list[ChatTurn] = []

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    def reset(self) -> None:
        """Clear conversation history."""

        self._history = []

    def history(self) -> list[dict[str, str]]:
        """Return conversation history as serializable dictionaries."""

        return [turn.model_dump() for turn in self._history]

    def load_history(self, history: list[dict[str, str]]) -> None:
        """Hydrate engine history from serialized chat turns."""

        self._history = []
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                self._history.append(ChatTurn(role=role, content=content))
        self._trim()

    def _trim(self) -> None:
        """Keep latest turns within configured history window."""

        excess = len(self._history) - self.max_turns * 2
        if excess > 0:
            self._history = self._history[excess:]

    def _to_chat_payload(self) -> list[dict[str, str]]:
        """Build chat payload with system prompt plus history."""

        payload = [{"role": "system", "content": self.system_prompt}]
        payload.extend(turn.model_dump() for turn in self._history)
        return payload

    def send(
        self, user_message: str, temperature: float = 0.3, max_tokens: int = 700
    ) -> ChatResult:
        """Add user message, run inference, append assistant response."""

        normalized = (user_message or "").strip()
        if not normalized:
            return ChatResult(
                response="",
                model=self.model,
                turns_kept=len(self._history),
                error=ErrorInfo(message="Message cannot be empty.", stage="validation"),
            )

        self._history.append(ChatTurn(role="user", content=normalized))
        self._trim()

        result = self.client.chat(
            model=self.model,
            messages=self._to_chat_payload(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if result["error"]:
            logger.error("Chat inference failed: %s", result["error"])
            return ChatResult(
                response="",
                model=self.model,
                turns_kept=len(self._history),
                latency_ms=result["latency_ms"],
                error=ErrorInfo(message=result["error"], stage="inference"),
            )

        assistant_text = result["response"].strip()
        self._history.append(ChatTurn(role="assistant", content=assistant_text))
        self._trim()

        return ChatResult(
            response=assistant_text,
            model=self.model,
            turns_kept=len(self._history),
            latency_ms=result["latency_ms"],
            error=None,
        )


def generate_reply(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 700,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    """Backwards-compatible stateless wrapper for UI/tests."""

    engine = ChatEngine(model=model, client=client)
    try:
        if not messages:
            return ChatResult(
                response="",
                model=engine.model,
                turns_kept=0,
                error=ErrorInfo(message="No messages provided.", stage="validation"),
            ).model_dump()

        prior_messages = messages[:-1]
        last = messages[-1]
        if last.get("role") != "user":
            return ChatResult(
                response="",
                model=engine.model,
                turns_kept=0,
                error=ErrorInfo(message="Last message must have role='user'.", stage="validation"),
            ).model_dump()

        engine.load_history(prior_messages)
        result = engine.send(
            last.get("content", ""), temperature=temperature, max_tokens=max_tokens
        )
        return result.model_dump()
    finally:
        engine.close()
