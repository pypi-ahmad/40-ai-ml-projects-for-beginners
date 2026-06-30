"""Memory search tool aliases."""

from __future__ import annotations

from crew_platform.tools.vector_search import SemanticSearchTool


class MemorySearchTool(SemanticSearchTool):
    name = "memory_search"
    description = "Search shared semantic memory"
