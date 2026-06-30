"""Vector search and semantic search over Chroma collections."""

from __future__ import annotations

from pydantic import BaseModel

from crew_platform.memory.chroma_store import ChromaSemanticStore
from crew_platform.tools.base import BaseTool


class VectorSearchInput(BaseModel):
    query: str
    top_k: int = 5


class VectorSearchOutput(BaseModel):
    matches: list[dict[str, str | float]]


class VectorSearchTool(BaseTool[VectorSearchInput, VectorSearchOutput]):
    name = "vector_search"
    description = "Semantic retrieval from Chroma memory"
    input_model = VectorSearchInput
    output_model = VectorSearchOutput

    def __init__(self, store: ChromaSemanticStore) -> None:
        self.store = store

    async def run(self, payload: VectorSearchInput) -> VectorSearchOutput:
        matches = self.store.search(payload.query, top_k=payload.top_k)
        return VectorSearchOutput(matches=matches)


class SemanticSearchTool(VectorSearchTool):
    name = "semantic_search"
    description = "Semantic search alias for vector retrieval"
