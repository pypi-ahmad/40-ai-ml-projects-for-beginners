"""Base protocol for MCP-style tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from task_planning_agent.schemas import ToolResult


class Tool(ABC):
    """Minimal tool abstraction."""

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute tool operation and return structured output."""
