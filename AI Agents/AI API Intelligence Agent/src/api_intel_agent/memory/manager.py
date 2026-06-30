"""Unified memory manager combining SQLite + Chroma stores."""

from __future__ import annotations

from api_intel_agent.core.schemas import AnalyzeResponse, MemorySearchHit, MemorySearchRequest
from api_intel_agent.memory.chroma_store import ChromaMemoryStore
from api_intel_agent.memory.sqlite_store import SQLiteMemoryStore


class MemoryManager:
    def __init__(self) -> None:
        self.sqlite = SQLiteMemoryStore()
        self.chroma = ChromaMemoryStore()

    def store_analysis(self, query: str, response: AnalyzeResponse) -> None:
        self.sqlite.save_response(query=query, response=response)
        self.chroma.upsert(response.run_id, response.summary, {"query": query, "status": response.status.value})
        for source in response.sources:
            self.sqlite.save_api_summary(response.run_id, source.provider, source.endpoint)

    def history(self, limit: int = 20):
        return self.sqlite.history(limit=limit)

    def get_response(self, run_id: str):
        return self.sqlite.get_response(run_id)

    def search(self, request: MemorySearchRequest) -> list[MemorySearchHit]:
        return self.chroma.search(request.query, top_k=request.top_k)
