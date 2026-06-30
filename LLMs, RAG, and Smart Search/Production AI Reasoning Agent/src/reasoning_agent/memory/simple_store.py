"""Simple in-memory store for short/working/tool memory."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque

from reasoning_agent.memory.base import MemoryEvent, MemoryHit, MemoryScope


class SimpleMemoryStore:
    """In-memory memory store with lexical retrieval."""

    def __init__(self, max_items_per_scope: int = 500) -> None:
        self.max_items = max_items_per_scope
        self._data: dict[MemoryScope, Deque[MemoryEvent]] = defaultdict(deque)

    def write(self, event: MemoryEvent) -> None:
        """Store event, dropping oldest when max is reached."""

        bucket = self._data[event.scope]
        bucket.append(event)
        while len(bucket) > self.max_items:
            bucket.popleft()

    def retrieve(self, query: str, k: int = 5, scope: MemoryScope | None = None) -> list[MemoryHit]:
        """Retrieve lexically ranked matches by token overlap."""

        tokens = {t.lower() for t in query.split() if t.strip()}
        events: list[MemoryEvent] = []
        if scope is None:
            for values in self._data.values():
                events.extend(list(values))
        else:
            events = list(self._data[scope])

        scored: list[MemoryHit] = []
        for event in events:
            etokens = {t.lower() for t in event.text.split() if t.strip()}
            overlap = len(tokens.intersection(etokens))
            if overlap <= 0:
                continue
            score = float(overlap) / float(max(len(tokens), 1))
            scored.append(MemoryHit(text=event.text, score=score, metadata=event.metadata))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]
