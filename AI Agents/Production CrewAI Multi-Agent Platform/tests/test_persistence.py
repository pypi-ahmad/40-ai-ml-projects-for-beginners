from __future__ import annotations

from pathlib import Path

from crew_platform.memory.persistence import PersistenceStore


def test_persistence_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "platform.db"
    store = PersistenceStore(str(db_path))

    store.save_plan("run-1", "session-a", "objective", {"tasks": []})
    store.save_task("run-1", "t1", "Analyst", "completed", {"x": 1}, None, 1)
    store.save_report("run-1", {"summary": "done"})
    store.save_tool_call("run-1", "calculator", {"expression": "1+1"}, "ok")
    store.save_conversation("run-1", "session-a", "user", "hello")

    tasks = store.fetch_tasks("run-1")
    report = store.fetch_report("run-1")
    memory = store.fetch_memory(limit=10)

    assert db_path.exists()
    assert len(tasks) == 1
    assert report and report["summary"] == "done"
    assert memory["conversations"]
