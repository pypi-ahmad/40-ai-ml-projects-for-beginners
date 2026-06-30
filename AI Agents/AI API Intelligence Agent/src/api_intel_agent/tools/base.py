"""Tool interfaces for agent tool-calling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    name: str
    success: bool
    payload: dict[str, Any]
    error: str | None = None


class Tool(Protocol):
    name: str

    async def run(self, **kwargs: Any) -> ToolResult: ...
