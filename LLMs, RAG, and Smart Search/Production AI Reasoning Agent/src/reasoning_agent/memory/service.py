"""Unified memory service orchestration."""

from __future__ import annotations

from reasoning_agent.memory.base import MemoryEvent, MemoryHit, MemoryScope
from reasoning_agent.memory.chroma_store import ChromaMemoryStore
from reasoning_agent.memory.simple_store import SimpleMemoryStore


class MemoryService:
    """Facade combining short-term and semantic memory stores."""

    def __init__(self, short_term: SimpleMemoryStore, semantic: ChromaMemoryStore) -> None:
        self.short_term = short_term
        self.semantic = semantic

    def write(self, event: MemoryEvent) -> None:
        """Write to both short-term and semantic stores."""

        self.short_term.write(event)
        self.semantic.write(event)

    def retrieve(self, query: str, k: int = 5, scope: MemoryScope | None = None) -> list[MemoryHit]:
        """Hybrid retrieve: semantic + lexical fallback."""

        semantic_hits = self.semantic.retrieve(query, k=k, scope=scope)
        if semantic_hits:
            return semantic_hits
        return self.short_term.retrieve(query, k=k, scope=scope)
