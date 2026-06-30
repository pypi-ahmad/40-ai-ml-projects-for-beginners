"""Runtime memory manager combining short-term and semantic memory."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from crew_platform.config import Settings
from crew_platform.memory.chroma_store import ChromaSemanticStore
from crew_platform.memory.session import SessionMemoryStore


class RuntimeMemory:
    """Shared runtime memory for active sessions and semantic retrieval."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = SessionMemoryStore(window_size=settings.memory.conversation_window)
        disable_chroma = os.getenv("CREW_PLATFORM_DISABLE_CHROMA", "0").lower() in {"1", "true", "yes"}
        if settings.memory.chroma_enabled and not disable_chroma:
            try:
                self.semantic_store = ChromaSemanticStore(
                    path=settings.memory.chroma_path,
                    collection_name="platform_memory",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chroma unavailable, semantic memory disabled: {}", exc)
                self.semantic_store = None
        else:
            self.semantic_store = None

    def append_conversation(self, role: str, content: str, run_id: str) -> None:
        self.session.append_event(role, content)
        if self.semantic_store is None:
            return

        item_id = self._build_id(run_id, role, content)
        metadata = {
            "role": role,
            "run_id": run_id,
            "type": "conversation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.semantic_store.add(item_id=item_id, text=content, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic add failed: {}", exc)

    def remember_run(self, run_id: str, objective: str, summary: str) -> None:
        if self.semantic_store is None:
            return
        text = f"Objective: {objective}\nSummary: {summary}"
        metadata = {
            "type": "report",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.semantic_store.add(item_id=self._build_id(run_id, "report", text), text=text, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic remember_run failed: {}", exc)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.semantic_store is None:
            return []
        k = top_k if top_k is not None else self.settings.memory.retrieval_top_k
        try:
            return self.semantic_store.search(query=query, top_k=k)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic search failed: {}", exc)
            return []

    @staticmethod
    def _build_id(run_id: str, role: str, content: str) -> str:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        return f"{run_id}-{role}-{digest}"
