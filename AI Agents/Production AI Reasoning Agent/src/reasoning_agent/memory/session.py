"""Session and short-term memory stores."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemoryEvent:
    role: str
    content: str


class SessionMemoryStore:
    """Windowed in-memory conversation store."""

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self._events: list[MemoryEvent] = []

    def append_event(self, role: str, content: str) -> None:
        self._events.append(MemoryEvent(role=role, content=content))
        if len(self._events) > self.window_size:
            self._events = self._events[-self.window_size :]

    def recent_context(self) -> list[MemoryEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
