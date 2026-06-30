"""Conversation memory with pruning and summarization hooks."""

from __future__ import annotations

from collections import deque

from hybrid_research_assistant.schemas import ConversationTurn


class ConversationMemory:
    """Short-term memory with rolling retention and summary placeholder."""

    def __init__(self, max_turns: int = 20, summary_trigger_turns: int = 12) -> None:
        self.max_turns = max_turns
        self.summary_trigger_turns = summary_trigger_turns
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.summary: str = ""

    def add(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))

    def history(self) -> list[ConversationTurn]:
        return list(self.turns)

    def maybe_prune(self) -> bool:
        """Prune old turns when memory length exceeds summary threshold."""

        if len(self.turns) < self.summary_trigger_turns:
            return False
        while len(self.turns) > max(4, self.max_turns // 2):
            self.turns.popleft()
        return True

    def set_summary(self, summary: str) -> None:
        self.summary = summary
