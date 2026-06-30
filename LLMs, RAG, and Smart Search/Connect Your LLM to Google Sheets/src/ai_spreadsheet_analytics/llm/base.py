"""LLM client protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class LLMResponse:
    """Standardized LLM response."""

    text: str
    latency_ms: float
    token_estimate: int
    raw: dict


class LLMClient(Protocol):
    """LLM client interface."""

    async def agenerate(self, model: str, prompt: str, system: str = "", temperature: float = 0.0) -> LLMResponse:
        """Generate completion asynchronously."""
