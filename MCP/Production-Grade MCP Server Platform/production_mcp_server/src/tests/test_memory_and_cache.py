from __future__ import annotations

import time
from pathlib import Path

from config.settings import load_settings
from memory.service import MemoryService


def test_sqlite_memory_and_cache(tmp_path: Path) -> None:
    settings = load_settings("configs/default.yaml")
    settings.memory.sqlite_path = str(tmp_path / "memory.db")

    memory = MemoryService(settings)
    memory.log_conversation(session_id="s1", role="user", content="hello")
    rows = memory.fetch_recent_conversations(session_id="s1", limit=5)
    assert len(rows) == 1
    assert rows[0]["content"] == "hello"

    memory.cache_set("k1", {"x": 1}, ttl_seconds=1)
    assert memory.cache_get("k1") == {"x": 1}
    time.sleep(1.1)
    assert memory.cache_get("k1") is None
