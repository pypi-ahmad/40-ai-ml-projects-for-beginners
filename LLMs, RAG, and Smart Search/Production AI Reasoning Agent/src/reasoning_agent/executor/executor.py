"""Tool execution orchestration."""

from __future__ import annotations

from typing import Any

from reasoning_agent.schemas import ToolObservation
from reasoning_agent.tooling import ToolContext, ToolRegistry


class Executor:
    """Invoke selected tool and normalize observation."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        *,
        session_id: str,
        run_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> ToolObservation:
        """Run one tool call."""

        ctx = ToolContext(
            session_id=session_id,
            run_id=run_id,
            workspace_root=self.registry.workspace_root,
        )
        result = self.registry.invoke(tool_name, tool_args, ctx)

        return ToolObservation(
            tool=tool_name,
            ok=result.ok,
            output=result.output,
            error=result.error,
            latency_ms=result.latency_ms,
            citations=result.citations,
        )
