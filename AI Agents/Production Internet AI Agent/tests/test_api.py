from __future__ import annotations

from typing import Any

import httpx
import pytest

from internet_agent.api.app import app


class FakeService:
    async def chat(self, session_id: str, message: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "query": message,
            "answer": "ok",
            "confidence": 0.9,
            "hallucination_risk": "low",
            "citations": [],
            "reasoning_trace": [],
            "tool_outputs": [],
            "report": {"query": message, "answer": "ok"},
        }

    async def search(self, session_id: str, query: str, providers=None) -> dict[str, Any]:
        return {
            "query": query,
            "providers": providers or ["duckduckgo"],
            "results": [],
            "documents": [],
            "chunks": [],
            "from_cache": False,
            "latency_ms": 1.0,
        }

    async def browse(self, session_id: str, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "content": "x",
            "markdown": "x",
            "status_code": 200,
            "content_type": "text/html",
        }

    def history(self, session_id: str) -> dict[str, Any]:
        return {"messages": [], "tool_history": [], "reports": []}

    def memory_search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        return {"query": query, "hits": []}

    def export_report(self, session_id: str, payload: dict[str, Any], fmt: str) -> dict[str, Any]:
        return {"format": fmt, "path": "outputs/reports/fake.json"}

    def metrics(self) -> dict[str, Any]:
        return {"counters": {}, "latencies": {}}

    def monitor(self) -> dict[str, Any]:
        return {"cpu_percent": 0.0, "memory": {"percent": 0.0}, "gpu": {"available": False}}

    def analytics(self, session_id: str | None = None) -> dict[str, Any]:
        return {"metrics": {}, "most_used_tools": []}


@pytest.mark.asyncio
async def test_api_endpoints(monkeypatch) -> None:
    monkeypatch.setattr("internet_agent.api.app.get_service", lambda: FakeService())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        assert health.status_code == 200

        chat = await client.post("/chat", json={"session_id": "s1", "message": "hello"})
        assert chat.status_code == 200
        assert chat.json()["answer"] == "ok"

        search = await client.post("/search", json={"session_id": "s1", "query": "python"})
        assert search.status_code == 200

        browse = await client.post("/browse", json={"session_id": "s1", "url": "https://example.com"})
        assert browse.status_code == 200

        memory = await client.post("/memory", json={"query": "x"})
        assert memory.status_code == 200
