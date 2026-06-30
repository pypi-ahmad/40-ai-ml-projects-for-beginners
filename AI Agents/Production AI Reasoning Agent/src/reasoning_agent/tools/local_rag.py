"""Local RAG helper tool built on document + vector search."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from reasoning_agent.tools.base import BaseTool
from reasoning_agent.tools.document_search import DocumentSearchTool, DocumentSearchInput


class LocalRAGInput(BaseModel):
    query: str


class LocalRAGOutput(BaseModel):
    context: str
    citations: list[str]


class LocalRAGTool(BaseTool[LocalRAGInput, LocalRAGOutput]):
    name = "local_rag"
    description = "Retrieve local document context for grounded answers"
    input_model = LocalRAGInput
    output_model = LocalRAGOutput

    def __init__(self, workspace_root: Path) -> None:
        self.search_tool = DocumentSearchTool(workspace_root)

    async def run(self, payload: LocalRAGInput) -> LocalRAGOutput:
        hits = await self.search_tool.run(DocumentSearchInput(query=payload.query, max_hits=3))
        context_parts = [hit["snippet"] for hit in hits.hits]
        citations = [hit["path"] for hit in hits.hits]
        return LocalRAGOutput(context="\n\n".join(context_parts), citations=citations)
