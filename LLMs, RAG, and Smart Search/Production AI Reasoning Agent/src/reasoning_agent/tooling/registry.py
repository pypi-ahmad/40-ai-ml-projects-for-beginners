"""Dynamic tool registry with validation and audit logging."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from reasoning_agent.observability.logger import get_logger, log_event
from reasoning_agent.tooling.base import ToolContext, ToolResult, ToolSpec

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for dynamic tool registration and invocation."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[BaseModel, ToolContext], BaseModel]] = {}

    def register(self, spec: ToolSpec, handler: Callable[[BaseModel, ToolContext], BaseModel]) -> None:
        """Register tool metadata and callable."""

        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler
        log_event(logger, "tool_registered", tool=spec.name, tags=spec.tags)

    def register_fn(self, spec: ToolSpec) -> Callable[[Callable[[BaseModel, ToolContext], BaseModel]], Callable[[BaseModel, ToolContext], BaseModel]]:
        """Decorator helper for registration."""

        def decorator(func: Callable[[BaseModel, ToolContext], BaseModel]) -> Callable[[BaseModel, ToolContext], BaseModel]:
            self.register(spec, func)
            return func

        return decorator

    def specs(self) -> list[ToolSpec]:
        """Return all tool specs."""

        return list(self._specs.values())

    def discover(self, query: str | None = None, tags: list[str] | None = None) -> list[ToolSpec]:
        """Discover tools by query and tags."""

        result = self.specs()
        if query:
            q = query.lower()
            result = [
                spec
                for spec in result
                if q in spec.name.lower() or q in spec.description.lower()
            ]
        if tags:
            tagset = {tag.lower() for tag in tags}
            result = [spec for spec in result if tagset.intersection({t.lower() for t in spec.tags})]
        return sorted(result, key=lambda spec: spec.name)

    def invoke(self, name: str, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        """Invoke tool with schema validation and audit logging."""

        if name not in self._specs:
            return ToolResult(ok=False, output=None, error=f"Unknown tool: {name}", latency_ms=0.0)

        spec = self._specs[name]
        handler = self._handlers[name]
        started = time.perf_counter()

        try:
            parsed_input = spec.input_model.model_validate(payload)
        except Exception as exc:
            return ToolResult(
                ok=False,
                output=None,
                error=f"Input validation failed for {name}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
                validation_passed=False,
            )

        try:
            out_model = handler(parsed_input, context)
            validated_out = spec.output_model.model_validate(out_model.model_dump())
            latency_ms = (time.perf_counter() - started) * 1000
            result = ToolResult(
                ok=True,
                output=validated_out.model_dump(),
                error=None,
                latency_ms=latency_ms,
            )
            log_event(
                logger,
                "tool_invocation",
                tool=name,
                ok=True,
                latency_ms=latency_ms,
                session_id=context.session_id,
                run_id=context.run_id,
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            log_event(
                logger,
                "tool_invocation",
                tool=name,
                ok=False,
                latency_ms=latency_ms,
                error=str(exc),
                session_id=context.session_id,
                run_id=context.run_id,
            )
            return ToolResult(ok=False, output=None, error=str(exc), latency_ms=latency_ms)

    def metadata(self) -> list[dict[str, Any]]:
        """Tool metadata for planner/router prompts."""

        data: list[dict[str, Any]] = []
        for spec in self.specs():
            data.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "tags": spec.tags,
                }
            )
        return data
