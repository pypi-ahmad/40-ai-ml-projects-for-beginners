"""Semantic search tool over memory store."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.memory import MemoryScope, MemoryService
from reasoning_agent.tooling.base import ToolContext, ToolSpec


class SemanticSearchInput(BaseModel):
    """Semantic search input payload."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class SemanticSearchOutput(BaseModel):
    """Semantic search output payload."""

    query: str
    hits: list[dict[str, object]]


def make_handler(memory: MemoryService):
    """Create semantic search handler."""

    def handler(payload: SemanticSearchInput, _: ToolContext) -> SemanticSearchOutput:
        rows = memory.retrieve(payload.query, k=payload.top_k, scope=MemoryScope.SEMANTIC)
        return SemanticSearchOutput(
            query=payload.query,
            hits=[
                {
                    "text": hit.text,
                    "score": hit.score,
                    "metadata": hit.metadata,
                }
                for hit in rows
            ],
        )

    return handler


spec = ToolSpec(
    name="semantic_search",
    description="Semantic retrieval from persistent memory",
    input_model=SemanticSearchInput,
    output_model=SemanticSearchOutput,
    tags=["memory", "semantic", "search"],
)
