from __future__ import annotations

from reasoning_agent.memory.manager import MemoryManager
from reasoning_agent.memory.session import SessionMemoryStore


class BrokenSemanticStore:
    def add(self, item_id: str, text: str, metadata: dict[str, str] | None = None) -> None:
        del item_id, text, metadata
        raise RuntimeError("offline")

    def search(self, query: str, top_k: int = 5):  # noqa: ANN001
        del query, top_k
        raise RuntimeError("search-down")


def test_memory_manager_serializes_recent_context() -> None:
    manager = MemoryManager(session=SessionMemoryStore(window_size=5), semantic=None)
    manager.append(role="user", content="hello", run_id="run-1")

    context = manager.context_for_query("hello")

    assert context["recent"] == [{"role": "user", "content": "hello"}]
    assert context["semantic"] == []


def test_memory_manager_handles_semantic_errors_gracefully() -> None:
    manager = MemoryManager(session=SessionMemoryStore(window_size=5), semantic=BrokenSemanticStore())
    manager.append(role="assistant", content="answer", run_id="run-2")

    context = manager.context_for_query("answer")

    assert context["recent"][0]["role"] == "assistant"
    assert context["semantic"] == []
