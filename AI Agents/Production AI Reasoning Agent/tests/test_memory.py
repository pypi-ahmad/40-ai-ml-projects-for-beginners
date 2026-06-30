from __future__ import annotations

from reasoning_agent.memory.session import SessionMemoryStore


def test_session_memory_append_and_retrieve() -> None:
    store = SessionMemoryStore(window_size=3)

    store.append_event("user", "one")
    store.append_event("assistant", "two")
    store.append_event("user", "three")
    store.append_event("assistant", "four")

    context = store.recent_context()

    assert len(context) == 3
    assert context[0].content == "two"
    assert context[-1].content == "four"
