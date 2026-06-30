"""RAG retriever service."""

from __future__ import annotations

from typing import Any

from crew_platform.config import Settings
from crew_platform.memory.runtime import RuntimeMemory


class RAGRetriever:
    """Retrieves semantic context from shared memory."""

    def __init__(self, settings: Settings, memory: RuntimeMemory) -> None:
        self.settings = settings
        self.memory = memory

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k if top_k is not None else self.settings.memory.retrieval_top_k
        return self.memory.search(query, top_k=k)
