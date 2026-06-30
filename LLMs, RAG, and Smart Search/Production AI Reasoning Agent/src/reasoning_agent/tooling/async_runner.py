"""Async/parallel tool invocation helper (phase-2 optional)."""

from __future__ import annotations

import asyncio
from typing import Any

from reasoning_agent.tooling import ToolContext, ToolRegistry


class AsyncToolRunner:
    """Run independent tools concurrently."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def invoke_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ToolContext,
    ) -> list[dict[str, Any]]:
        """Run tool calls in parallel with asyncio threads."""

        async def run_one(name: str, payload: dict[str, Any]) -> dict[str, Any]:
            result = await asyncio.to_thread(self.registry.invoke, name, payload, context)
            return {
                "tool": name,
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
                "latency_ms": result.latency_ms,
            }

        tasks = [run_one(name, payload) for name, payload in calls]
        return await asyncio.gather(*tasks)
