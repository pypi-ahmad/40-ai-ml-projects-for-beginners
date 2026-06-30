"""Memory contracts for agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class MemoryScope(str, Enum):
    """Memory scope categories."""

    CONVERSATION = "conversation"
    WORKING = "working"
    TOOL = "tool"
    OBSERVATION = "observation"
    SEMANTIC = "semantic"


@dataclass(slots=True)
class MemoryEvent:
    """Normalized memory write payload."""

    session_id: str
    run_id: str
    scope: MemoryScope
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class MemoryHit:
    """Retrieved memory record."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    """Memory persistence interface."""

    def write(self, event: MemoryEvent) -> None:
        """Persist memory event."""

    def retrieve(self, query: str, k: int = 5, scope: MemoryScope | None = None) -> list[MemoryHit]:
        """Retrieve relevant memory hits."""
