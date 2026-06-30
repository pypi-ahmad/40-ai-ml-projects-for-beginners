"""Observation processing and memory writes."""

from __future__ import annotations

from reasoning_agent.memory import MemoryEvent, MemoryScope, MemoryService
from reasoning_agent.schemas import ToolObservation


class ObservationProcessor:
    """Persist observations and create summary text."""

    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory

    def process(self, *, session_id: str, run_id: str, observation: ToolObservation) -> str:
        """Write observation to memory and return summary string."""

        text = (
            f"tool={observation.tool} ok={observation.ok} "
            f"output={observation.output} error={observation.error}"
        )
        scope = MemoryScope.OBSERVATION if observation.ok else MemoryScope.TOOL
        self.memory.write(
            MemoryEvent(
                session_id=session_id,
                run_id=run_id,
                scope=scope,
                text=text,
                metadata={"tool": observation.tool, "ok": observation.ok},
            )
        )

        self.memory.write(
            MemoryEvent(
                session_id=session_id,
                run_id=run_id,
                scope=MemoryScope.SEMANTIC,
                text=text,
                metadata={"tool": observation.tool},
            )
        )
        return text
