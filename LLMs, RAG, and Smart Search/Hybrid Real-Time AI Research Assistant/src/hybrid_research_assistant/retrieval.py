"""Local/web/hybrid retrieval and routing policies."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from hybrid_research_assistant.embeddings import EmbeddingProvider
from hybrid_research_assistant.schemas import RetrievalMode, RetrievedContext, RouteDecision
from hybrid_research_assistant.vectordb import ChromaVectorStore
from hybrid_research_assistant.web_search import WebSearchService


class IntentRouter:
    """Heuristic-first query router for local/web/hybrid modes."""

    def route(self, query: str, requested_mode: RetrievalMode) -> RouteDecision:
        if requested_mode != RetrievalMode.AUTO:
            return RouteDecision(mode=requested_mode, reason="user_selected_mode", confidence=1.0)

        lowered = query.lower()
        fresh_cues = ["today", "yesterday", "latest", "news", "announcement", "recent", "this week"]
        compare_cues = ["compare", "versus", "vs", "difference", "policy"]

        has_fresh = any(cue in lowered for cue in fresh_cues)
        has_compare = any(cue in lowered for cue in compare_cues)

        if has_fresh and has_compare:
            return RouteDecision(mode=RetrievalMode.HYBRID, reason="fresh_and_comparative_query", confidence=0.8)
        if has_fresh:
            return RouteDecision(mode=RetrievalMode.WEB, reason="freshness_detected", confidence=0.85)
        if has_compare:
            return RouteDecision(mode=RetrievalMode.HYBRID, reason="comparative_query_detected", confidence=0.75)
        return RouteDecision(mode=RetrievalMode.LOCAL, reason="default_local_knowledge", confidence=0.7)


class RetrievalService:
    """Unified retrieval service across local and web sources."""

    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore,
        embedder: EmbeddingProvider,
        web_search: WebSearchService,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.web_search = web_search

    def retrieve_local(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedContext], float]:
        started = time.perf_counter()
        query_embedding = self.embedder.embed_texts([query])[0]
        rows = self.vector_store.query(query_embedding=query_embedding, top_k=top_k, where=metadata_filter)
        latency_ms = (time.perf_counter() - started) * 1000
        return rows, latency_ms

    async def retrieve_web(
        self,
        query: str,
        *,
        top_k: int,
        provider: str | None = None,
    ) -> tuple[list[RetrievedContext], float]:
        started = time.perf_counter()
        rows = await self.web_search.search(query, k=top_k, provider=provider)
        contexts = [
            RetrievedContext(
                chunk_id=f"web::{index}::{row.provider}",
                doc_id=f"web::{row.url}",
                text=row.snippet,
                score=row.score,
                metadata={
                    "source": row.url,
                    "source_type": "web",
                    "url": row.url,
                    "document_title": row.title,
                    "provider": row.provider,
                    "page_number": None,
                },
                source="web",
            )
            for index, row in enumerate(rows, start=1)
        ]
        latency_ms = (time.perf_counter() - started) * 1000
        return contexts, latency_ms

    async def retrieve_hybrid(
        self,
        query: str,
        *,
        local_k: int,
        web_k: int,
        metadata_filter: dict[str, Any] | None = None,
        provider: str | None = None,
    ) -> tuple[list[RetrievedContext], float]:
        started = time.perf_counter()
        local_task = asyncio.to_thread(
            self.retrieve_local,
            query,
            top_k=local_k,
            metadata_filter=metadata_filter,
        )
        web_task = self.retrieve_web(query, top_k=web_k, provider=provider)
        local_result, web_result = await asyncio.gather(local_task, web_task)
        local_rows, _local_ms = local_result
        web_rows, _web_ms = web_result
        merged = self.merge_and_deduplicate(local_rows + web_rows)
        latency_ms = (time.perf_counter() - started) * 1000
        return merged, latency_ms

    @staticmethod
    def merge_and_deduplicate(rows: list[RetrievedContext]) -> list[RetrievedContext]:
        """Merge sources and dedupe by url/source/chunk hash."""

        unique: dict[str, RetrievedContext] = {}
        for row in rows:
            url = str(row.metadata.get("url", ""))
            source = str(row.metadata.get("source", ""))
            key = url or source or row.chunk_id
            existing = unique.get(key)
            if existing is None or row.score > existing.score:
                unique[key] = row
        return sorted(unique.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def mmr_select(rows: list[RetrievedContext], top_k: int, lambda_mult: float = 0.5) -> list[RetrievedContext]:
        """Approximate MMR selection from scored candidates."""

        if not rows or top_k <= 0:
            return []
        selected: list[RetrievedContext] = []
        remaining = sorted(rows, key=lambda item: item.score, reverse=True)
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_idx = 0
            best_value = float("-inf")
            for idx, candidate in enumerate(remaining):
                relevance = candidate.score
                diversity_penalty = max(
                    (1.0 if candidate.doc_id == chosen.doc_id else 0.0 for chosen in selected),
                    default=0.0,
                )
                value = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)
                if value > best_value:
                    best_value = value
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected
