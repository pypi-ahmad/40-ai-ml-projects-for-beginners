"""End-to-end retrieval pipeline: search -> fetch -> extract -> chunk -> embed -> store."""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from internet_agent.config import Settings
from internet_agent.memory.chroma_store import ChromaMemoryStore
from internet_agent.memory.repository import MemoryRepository
from internet_agent.metrics import METRICS
from internet_agent.retrieval.chunking import chunk_text
from internet_agent.retrieval.extract import clean_html, extract_markdown, read_pdf_bytes
from internet_agent.retrieval.fetch import WebsiteFetcher
from internet_agent.retrieval.ranker import rank_sources
from internet_agent.retrieval.search import SearchProviders


class RetrievalPipeline:
    """Hybrid retrieval pipeline with cache + semantic memory + live web search."""

    def __init__(
        self,
        settings: Settings,
        memory_repo: MemoryRepository,
        semantic_store: ChromaMemoryStore,
    ) -> None:
        self.settings = settings
        self.memory_repo = memory_repo
        self.semantic_store = semantic_store
        self.search = SearchProviders(settings)
        self.fetcher = WebsiteFetcher(timeout_seconds=settings.search.request_timeout_seconds)

    async def run(
        self,
        session_id: str,
        query: str,
        providers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run full retrieval path and return ranked documents and chunks."""

        started = time.perf_counter()
        METRICS.inc("retrieval.active_searches", 1)
        cache_key = self.memory_repo.make_cache_key("search", query)
        try:
            if self.settings.cache.enabled:
                cached = self.memory_repo.cache_get(cache_key)
                if cached:
                    METRICS.inc("cache.search.hit")
                    return {**cached, "from_cache": True}
            METRICS.inc("cache.search.miss")

            semantic_hits = self.semantic_store.query(
                query, top_k=self.settings.memory.memory_top_k
            )

            provider_results = await self.search.search_all(
                query=query,
                providers=providers,
                max_results=self.settings.search.default_max_results,
            )

            flattened = [item for rows in provider_results.values() for item in rows if not item.get("error")]
            ranked = rank_sources(flattened, query)[: self.settings.retrieval.max_urls_per_query]

            fetched_docs = await self._fetch_and_extract(session_id=session_id, rows=ranked)
            chunks = self._chunk_documents(fetched_docs)

            semantic_ids = []
            semantic_texts = []
            semantic_meta = []
            for chunk in chunks:
                chunk_id = hashlib.sha256(
                    f"{chunk['url']}::{chunk['index']}::{chunk['content'][:100]}".encode()
                ).hexdigest()
                semantic_ids.append(chunk_id)
                semantic_texts.append(chunk["content"])
                semantic_meta.append(
                    {"url": chunk["url"], "title": chunk["title"], "source": chunk["source"]}
                )

            if semantic_ids:
                embed_started = time.perf_counter()
                self.semantic_store.upsert(
                    ids=semantic_ids, texts=semantic_texts, metadatas=semantic_meta
                )
                METRICS.observe_ms(
                    "retrieval.embedding.latency_ms", (time.perf_counter() - embed_started) * 1000
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            METRICS.observe_ms("retrieval.pipeline.latency_ms", elapsed_ms)
            METRICS.inc("retrieval.pipeline.runs")

            result = {
                "query": query,
                "providers": list(provider_results.keys()),
                "semantic_hits": semantic_hits,
                "results": ranked,
                "documents": fetched_docs,
                "chunks": chunks,
                "from_cache": False,
                "latency_ms": elapsed_ms,
            }

            if self.settings.cache.enabled:
                self.memory_repo.cache_set(
                    key=cache_key,
                    value=result,
                    ttl_seconds=self.settings.cache.default_ttl_seconds,
                    namespace="search",
                )

            return result
        finally:
            METRICS.inc("retrieval.active_searches", -1)

    async def _fetch_and_extract(self, session_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        async def _worker(row: dict[str, Any]) -> dict[str, Any] | None:
            url = row.get("url", "")
            if not url:
                return None
            try:
                payload = await self.fetcher.fetch(url)
            except Exception as exc:  # noqa: BLE001
                return {
                    "url": url,
                    "title": row.get("title", ""),
                    "source": row.get("source", "web"),
                    "content": "",
                    "markdown": "",
                    "error": str(exc),
                }

            content_type = payload.get("content_type", "")
            raw_text = ""
            markdown = ""
            if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                raw_text = read_pdf_bytes(payload["bytes"])
                markdown = raw_text
            else:
                raw_text = clean_html(payload["text"])
                markdown = extract_markdown(payload["text"])

            raw_text = raw_text[: self.settings.retrieval.max_content_chars]
            markdown = markdown[: self.settings.retrieval.max_content_chars]

            doc = {
                "url": url,
                "title": row.get("title", ""),
                "source": row.get("source", "web"),
                "status_code": payload.get("status_code", 0),
                "content": raw_text,
                "markdown": markdown,
            }
            self.memory_repo.add_visited_url(
                session_id=session_id,
                url=url,
                title=row.get("title", ""),
                status_code=payload.get("status_code", 0),
            )
            self.memory_repo.add_document(
                session_id=session_id,
                url=url,
                title=row.get("title", ""),
                content=raw_text,
                metadata={"source": row.get("source", "web")},
            )
            return doc

        docs = await asyncio.gather(*[_worker(row) for row in rows])
        return [doc for doc in docs if doc and doc.get("content")]

    def _chunk_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for doc in docs:
            pieces = chunk_text(
                text=doc.get("content", ""),
                chunk_size=self.settings.retrieval.chunk_size,
                chunk_overlap=self.settings.retrieval.chunk_overlap,
            )
            for index, piece in enumerate(pieces):
                out.append(
                    {
                        "url": doc.get("url", ""),
                        "title": doc.get("title", ""),
                        "source": doc.get("source", "web"),
                        "index": index,
                        "content": piece,
                    }
                )
        return out
