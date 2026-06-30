"""Plugin contracts for nodes/tools/workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class NodePlugin(Protocol):
    """Contract for registering new nodes."""

    name: str

    def register(self, workflow_builder: object) -> None:
        """Register plugin node into workflow builder."""


class ToolPlugin(Protocol):
    """Contract for registering new tools."""

    name: str

    def register(self, tool_registry: object) -> None:
        """Register plugin tool into tool registry."""


class WorkflowPlugin(Protocol):
    """Contract for custom workflow templates."""

    name: str

    def register(self, workflow_registry: object) -> None:
        """Register workflow plugin."""


@dataclass(slots=True)
class PluginBundle:
    """Container for plugin components."""

    name: str
    node_plugins: list[NodePlugin]
    tool_plugins: list[ToolPlugin]
    workflow_plugins: list[WorkflowPlugin]
