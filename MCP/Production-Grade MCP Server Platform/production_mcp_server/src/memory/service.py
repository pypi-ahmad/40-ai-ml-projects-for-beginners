from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from config.settings import Settings
from memory.chroma_store import ChromaStore
from memory.sqlite_store import SQLiteStore


class MemoryService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sqlite = SQLiteStore(settings.memory.sqlite_path)
        Path(settings.memory.chroma_path).mkdir(parents=True, exist_ok=True)
        self.chroma = ChromaStore(
            settings.memory.chroma_path,
            enabled=settings.memory.chroma_enabled,
        )

    def log_conversation(self, session_id: str, role: str, content: str) -> None:
        self.sqlite.log_conversation(session_id=session_id, role=role, content=content)

    def fetch_recent_conversations(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        return self.sqlite.fetch_recent_conversations(session_id=session_id, limit=limit)

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        latency_ms: int,
    ) -> None:
        self.sqlite.log_tool_call(session_id, tool_name, request_payload, response_payload, latency_ms)

    def log_prompt(self, prompt_name: str, variables: dict[str, Any], rendered_prompt: str) -> None:
        self.sqlite.log_prompt(prompt_name, variables, rendered_prompt)

    def log_response(self, session_id: str, response_type: str, payload: dict[str, Any]) -> None:
        self.sqlite.log_response(session_id, response_type, payload)

    def log_audit(self, actor: str, action: str, payload: dict[str, Any]) -> None:
        self.sqlite.log_audit(actor, action, payload)

    def log_usage(self, event_type: str, payload: dict[str, Any]) -> None:
        self.sqlite.log_usage(event_type, payload)

    def log_metric(self, metric_name: str, metric_value: float, labels: dict[str, Any]) -> None:
        self.sqlite.log_metric(metric_name, metric_value, labels)

    def recent_metrics(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.sqlite.recent_metrics(limit=limit)

    def cache_set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.settings.cache.ttl_seconds
        self.sqlite.cache_set(key, value, ttl)

    def cache_get(self, key: str) -> dict[str, Any] | None:
        return self.sqlite.cache_get(key)

    def cache_cleanup(self) -> int:
        return self.sqlite.cache_cleanup()

    def store_semantic_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        if not self.chroma.available:
            return ""
        identifier = doc_id or str(uuid4())
        self.chroma.upsert([{"id": identifier, "text": text, "metadata": metadata or {}}])
        return identifier

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = top_k if top_k is not None else self.settings.memory.top_k
        return [
            {
                "id": item.id,
                "text": item.text,
                "score": item.score,
                "metadata": item.metadata,
            }
            for item in self.chroma.search(query, top_k=limit)
        ]
