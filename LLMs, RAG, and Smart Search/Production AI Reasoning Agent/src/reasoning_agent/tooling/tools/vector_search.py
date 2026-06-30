"""Vector search tool with scope filtering."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reasoning_agent.memory import MemoryScope, MemoryService
from reasoning_agent.tooling.base import ToolContext, ToolSpec


class VectorSearchInput(BaseModel):
    """Vector search input payload."""

    query: str
    scope: str = Field(default="semantic")
    top_k: int = Field(default=5, ge=1, le=20)


class VectorSearchOutput(BaseModel):
    """Vector search output payload."""

    query: str
    scope: str
    hits: list[dict[str, object]]


def make_handler(memory: MemoryService):
    """Create vector search handler."""

    def handler(payload: VectorSearchInput, _: ToolContext) -> VectorSearchOutput:
        scope = MemoryScope(payload.scope)
        rows = memory.retrieve(payload.query, k=payload.top_k, scope=scope)
        return VectorSearchOutput(
            query=payload.query,
            scope=payload.scope,
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
    name="vector_search",
    description="Vector search over scoped memory collections",
    input_model=VectorSearchInput,
    output_model=VectorSearchOutput,
    tags=["memory", "vector", "search"],
)
