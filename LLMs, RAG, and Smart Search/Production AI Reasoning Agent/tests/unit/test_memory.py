from __future__ import annotations

from reasoning_agent.memory import MemoryEvent, MemoryScope, SimpleMemoryStore


def test_simple_memory_retrieval() -> None:
    store = SimpleMemoryStore()
    store.write(
        MemoryEvent(
            session_id="s",
            run_id="r",
            scope=MemoryScope.CONVERSATION,
            text="Agent used calculator to compute result",
        )
    )
    hits = store.retrieve("calculator result", k=3)
    assert hits
    assert hits[0].score > 0.0
