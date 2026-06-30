from __future__ import annotations

from internet_agent.config import Settings
from internet_agent.memory.repository import MemoryRepository


def _settings(sqlite_url: str) -> Settings:
    settings = Settings()
    settings.memory.sqlite_url = sqlite_url
    settings.memory.chroma_path = "artifacts/chroma-test"
    return settings


def test_cache_set_get(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    repo = MemoryRepository(_settings(f"sqlite:///{db_path}"))

    key = repo.make_cache_key("search", "python")
    repo.cache_set(key, {"value": 1}, ttl_seconds=60, namespace="search")
    found = repo.cache_get(key)
    assert found == {"value": 1}
