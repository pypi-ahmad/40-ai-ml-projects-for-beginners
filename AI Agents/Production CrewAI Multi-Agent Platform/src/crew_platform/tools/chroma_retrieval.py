"""Chroma retrieval alias tool."""

from __future__ import annotations

from crew_platform.tools.vector_search import VectorSearchTool


class ChromaRetrievalTool(VectorSearchTool):
    name = "chroma_retrieval"
    description = "Retrieve semantic context from Chroma collection"
