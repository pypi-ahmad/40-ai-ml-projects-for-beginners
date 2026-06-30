"""Tool execution engine."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState, ToolExecution
from reasoning_agent.tools.registry import ToolRegistry


@dataclass(slots=True)
class Executor:
    registry: ToolRegistry

    async def execute_current_step(self, state: AgentState, run_id: str) -> AgentState:
        current = next((step for step in state.plan if step.status in {"ready", "ready_no_tool"}), None)
        if current is None:
            state.done = True
            return state

        if current.status == "ready_no_tool" or current.tool_name is None:
            current.status = "completed"
            state.observations.append(current.description)
            return state

        try:
            output = await self.registry.invoke(current.tool_name, current.tool_input, run_id=run_id)
            current.status = "completed"
            state.tool_calls.append(
                ToolExecution(
                    tool_name=current.tool_name,
                    input_payload=current.tool_input,
                    output_payload=output,
                    success=True,
                )
            )
            state.observations.append(f"{current.tool_name}: {output}")
            return state
        except Exception as exc:  # noqa: BLE001
            current.status = "failed"
            state.tool_calls.append(
                ToolExecution(
                    tool_name=current.tool_name,
                    input_payload=current.tool_input,
                    output_payload={},
                    success=False,
                    error=str(exc),
                )
            )
            state.error = str(exc)
            return state
