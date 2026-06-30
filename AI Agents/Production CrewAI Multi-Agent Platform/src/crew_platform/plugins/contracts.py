"""Plugin contracts for agents, tools, workflows."""

from __future__ import annotations

from typing import Protocol


class AgentPlugin(Protocol):
    def register_agents(self) -> list[dict]:
        """Return additional agent definitions."""


class ToolPlugin(Protocol):
    def register_tools(self) -> list[str]:
        """Return tool module paths or names."""


class WorkflowPlugin(Protocol):
    def register_workflows(self) -> list[dict]:
        """Return workflow descriptors."""
