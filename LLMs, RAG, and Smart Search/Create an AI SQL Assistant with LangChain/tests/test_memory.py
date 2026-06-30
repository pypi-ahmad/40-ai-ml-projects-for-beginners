from __future__ import annotations

from pathlib import Path

from ai_sql_assistant.memory.store import AppStateStore


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    store = AppStateStore(tmp_path / "state.db")

    store.add_history(
        conversation_id="conv1",
        user_id="user1",
        question="q1",
        sql="SELECT 1",
        approach="direct",
        model="qwen3.5:4b",
        status="success",
        latency_ms=12.3,
        row_count=1,
    )
    store.add_turn("conv1", "user1", "q1", "SELECT 1", "explain")
    store.add_favorite("fav", "q1", "SELECT 1")

    history = store.history(limit=10)
    favorites = store.list_favorites()
    context = store.conversation_context("conv1")

    assert len(history) == 1
    assert len(favorites) == 1
    assert "Q: q1" in context
    assert "SQL: SELECT 1" in context
