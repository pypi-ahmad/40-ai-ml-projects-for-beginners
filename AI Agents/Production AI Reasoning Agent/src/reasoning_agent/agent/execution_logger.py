"""Execution trace logger."""

from __future__ import annotations

from dataclasses import dataclass

from reasoning_agent.agent.state import AgentState
from reasoning_agent.observability.events import EventRecord
from reasoning_agent.observability.tracer import JsonlTracer


@dataclass(slots=True)
class ExecutionLogger:
    tracer: JsonlTracer | None

    def log_state(self, run_id: str, node: str, state: AgentState) -> None:
        if self.tracer is None:
            return
        self.tracer.safe_log(
            EventRecord(
                event_type="node_transition",
                run_id=run_id,
                status="info",
                payload={
                    "node": node,
                    "iteration": state.iteration,
                    "done": state.done,
                    "error": state.error,
                    "plan": [step.model_dump(mode="json") for step in state.plan],
                },
            )
        )
