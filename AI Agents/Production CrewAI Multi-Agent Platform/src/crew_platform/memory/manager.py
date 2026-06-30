"""Combined memory manager."""

from __future__ import annotations

from datetime import datetime, timezone

from crew_platform.memory.chroma_store import ChromaSemanticStore
from crew_platform.memory.session import SessionMemoryStore


class MemoryManager:
    """Manages short-term and semantic memory tiers."""

    def __init__(self, session: SessionMemoryStore, semantic: ChromaSemanticStore | None = None) -> None:
        self.session = session
        self.semantic = semantic

    def append(self, role: str, content: str, run_id: str) -> None:
        self.session.append_event(role, content)
        if self.semantic is not None:
            event_id = f"{run_id}-{datetime.now(timezone.utc).timestamp()}"
            try:
                self.semantic.add(event_id, content, metadata={"role": role})
            except Exception:  # noqa: BLE001
                # Degrade gracefully when embedding model or storage backend is unavailable.
                pass

    def context_for_query(self, query: str, top_k: int = 5) -> dict[str, object]:
        recent = [
            {"role": event.role, "content": event.content} for event in self.session.recent_context()
        ]
        if self.semantic is None:
            semantic = []
        else:
            try:
                semantic = self.semantic.search(query, top_k=top_k)
            except Exception:  # noqa: BLE001
                semantic = []
        return {"recent": recent, "semantic": semantic}
