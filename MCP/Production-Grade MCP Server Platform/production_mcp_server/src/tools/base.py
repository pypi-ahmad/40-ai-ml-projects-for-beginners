from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    examples: list[dict[str, Any]] = field(default_factory=list)
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True
    open_world: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "examples": self.examples,
            "annotations": {
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.idempotent,
                "openWorldHint": self.open_world,
            },
        }
