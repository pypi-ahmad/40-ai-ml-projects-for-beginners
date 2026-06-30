"""Base contracts for tool system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel


class ToolExecutionError(RuntimeError):
    """Tool execution failure."""


@dataclass(slots=True)
class ToolContext:
    """Execution context passed to tool handlers."""

    session_id: str
    run_id: str
    workspace_root: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolSpec:
    """Tool metadata and schemas."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    tags: list[str] = field(default_factory=list)
    timeout_s: int = 20
    requires_network: bool = False
    required_env: list[str] = field(default_factory=list)

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON schema for input payload."""

        return self.input_model.model_json_schema()

    @property
    def output_schema(self) -> dict[str, Any]:
        """JSON schema for output payload."""

        return self.output_model.model_json_schema()


@dataclass(slots=True)
class ToolResult:
    """Normalized tool output envelope."""

    ok: bool
    output: dict[str, Any] | None
    error: str | None
    latency_ms: float
    citations: list[str] = field(default_factory=list)
    validation_passed: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ToolHandler(Protocol):
    """Protocol for tool handler callables."""

    def __call__(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        """Run tool and return output model."""
