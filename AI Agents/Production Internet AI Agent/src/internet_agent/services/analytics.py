"""Analytics and usage summarization."""

from __future__ import annotations

from collections import Counter

from internet_agent.memory.repository import MemoryRepository
from internet_agent.metrics import METRICS


def build_analytics(memory_repo: MemoryRepository, session_id: str | None = None) -> dict:
    snapshot = METRICS.snapshot()

    if session_id:
        history = memory_repo.get_tool_history(session_id=session_id, limit=500)
        tools = Counter(row["tool_name"] for row in history)
    else:
        tools = Counter()

    return {
        "metrics": snapshot,
        "most_used_tools": tools.most_common(15),
    }
