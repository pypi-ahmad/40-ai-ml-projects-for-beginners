"""Web search alias with canonical enterprise name."""

from __future__ import annotations

from crew_platform.tools.search import DuckDuckGoSearchTool


class WebSearchTool(DuckDuckGoSearchTool):
    name = "web_search"
    description = "Web search tool alias"
