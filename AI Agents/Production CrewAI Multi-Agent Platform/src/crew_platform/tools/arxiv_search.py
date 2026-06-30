"""Optional arXiv search tool."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

from crew_platform.tools.base import BaseTool
from crew_platform.utils.network import offline_mode


class ArxivSearchInput(BaseModel):
    query: str
    max_results: int = 5


class ArxivSearchOutput(BaseModel):
    papers: list[dict[str, str]]


class ArxivSearchTool(BaseTool[ArxivSearchInput, ArxivSearchOutput]):
    name = "arxiv_search"
    description = "Searches arXiv papers"
    input_model = ArxivSearchInput
    output_model = ArxivSearchOutput

    async def run(self, payload: ArxivSearchInput) -> ArxivSearchOutput:
        papers: list[dict[str, str]] = []
        if offline_mode():
            return ArxivSearchOutput(
                papers=[{"title": "offline-fallback", "summary": "arxiv disabled by AGENT_OFFLINE_MODE", "url": ""}]
            )
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": payload.query,
                        "start": 0,
                        "max_results": payload.max_results,
                    },
                )
                response.raise_for_status()

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                link = ""
                for node in entry.findall("atom:link", ns):
                    href = node.attrib.get("href")
                    if href:
                        link = href
                        break
                papers.append({"title": title, "summary": summary[:600], "url": link})
        except Exception as exc:  # noqa: BLE001
            papers.append({"title": "offline-fallback", "summary": f"arxiv unavailable: {exc}", "url": ""})
        return ArxivSearchOutput(papers=papers)
