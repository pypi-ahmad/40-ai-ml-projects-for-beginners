"""ChromaDB retrieval engine for local RAG pipeline."""

from __future__ import annotations

import logging
import math
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from rag_system.embeddings import EmbeddingEngine
from rag_system.types import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Primary vector retrieval engine backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "rag_documents_v1",
        persist_directory: str | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        default_top_k: int = 6,
    ) -> None:
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.default_top_k = default_top_k

        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, collection_name: str) -> Collection:
        """Create collection if missing, otherwise load existing one."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        """Reset collection state."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection(self.collection_name)

    def index_chunks(self, chunks: list[ChunkRecord], batch_size: int = 128) -> int:
        """Index chunk records into Chroma collection."""
        if not chunks:
            return 0

        indexed = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.text for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]
            metadatas = []
            for chunk in batch:
                metadata = {**chunk.metadata, "doc_id": chunk.doc_id}
                if chunk.parent_id is not None:
                    metadata["parent_id"] = chunk.parent_id
                metadatas.append(metadata)

            embeddings = self.embedding_engine.embed_batch(texts, batch_size=min(32, batch_size))

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            indexed += len(batch)

        logger.info("Indexed %d chunks in collection %s", indexed, self.collection_name)
        return indexed

    def query(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        include_parents: bool = False,
        dedupe_by_doc: bool = False,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k relevant chunks with normalized scores."""
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed(query)
        n_results = min(top_k or self.default_top_k, self.collection.count())

        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for idx, chunk_id in enumerate(ids):
            distance = float(dists[idx]) if idx < len(dists) else math.inf
            # Chroma cosine distance is roughly in [0, 2]. Convert to score [0, 1].
            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            metadata = metas[idx] if idx < len(metas) and metas[idx] is not None else {}
            text = docs[idx] if idx < len(docs) else ""
            doc_id = str(metadata.get("doc_id", "unknown"))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    score=score,
                    distance=distance,
                    metadata=metadata,
                )
            )

        if include_parents:
            retrieved = self._expand_with_parent_chunks(retrieved)
        if dedupe_by_doc:
            retrieved = self._dedupe_results_by_doc(retrieved)

        return retrieved

    def multi_query(
        self,
        queries: list[str],
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Run multiple queries and merge with reciprocal rank fusion."""
        per_query_results = [self.query(q, top_k=top_k, metadata_filter=metadata_filter) for q in queries]

        # Reciprocal Rank Fusion gives robust merged ranking for multi-query retrieval.
        k_constant = 60.0
        fused_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for results in per_query_results:
            for rank, chunk in enumerate(results, start=1):
                fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (1.0 / (k_constant + rank))
                chunk_map[chunk.chunk_id] = chunk

        merged = list(chunk_map.values())
        merged.sort(key=lambda chunk: fused_scores.get(chunk.chunk_id, 0.0), reverse=True)

        # Attach fused score for downstream score analysis while preserving raw score too.
        for chunk in merged:
            chunk.metadata["rrf_score"] = fused_scores.get(chunk.chunk_id, 0.0)

        return merged[: (top_k or self.default_top_k)]

    def format_context(self, chunks: list[RetrievedChunk], max_chars: int = 6000) -> tuple[str, list[str]]:
        """Format retrieved chunks into citation-friendly prompt context."""
        context_blocks: list[str] = []
        citations: list[str] = []
        consumed = 0

        for idx, chunk in enumerate(chunks, start=1):
            title = str(chunk.metadata.get("title", "unknown"))
            citation = f"[{idx}] doc={chunk.doc_id} title={title} score={chunk.score:.3f}"
            block = f"{citation}\n{chunk.text}"
            if consumed + len(block) > max_chars:
                break
            context_blocks.append(block)
            citations.append(citation)
            consumed += len(block)

        if not context_blocks:
            return "No relevant context retrieved.", []
        return "\n\n".join(context_blocks), citations

    def score_summary(self, chunks: list[RetrievedChunk]) -> dict[str, float]:
        """Summarize score distribution for observability charts."""
        if not chunks:
            return {
                "count": 0.0,
                "mean_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
            }

        scores = [chunk.score for chunk in chunks]
        return {
            "count": float(len(scores)),
            "mean_score": float(sum(scores) / len(scores)),
            "max_score": float(max(scores)),
            "min_score": float(min(scores)),
        }

    def get_collection_stats(self) -> dict[str, Any]:
        """Return lightweight collection diagnostics."""
        snapshot = self.collection.get(include=["metadatas"], limit=min(2000, max(1, self.collection.count())))
        metadatas = snapshot.get("metadatas", []) or []
        unique_docs = {str(meta.get("doc_id")) for meta in metadatas if meta and meta.get("doc_id")}
        strategies = {}
        for meta in metadatas:
            if not meta:
                continue
            strategy = str(meta.get("strategy", "unknown"))
            strategies[strategy] = strategies.get(strategy, 0) + 1
        return {
            "collection_name": self.collection_name,
            "count": self.collection.count(),
            "sampled_unique_docs": len(unique_docs),
            "strategy_distribution_sample": strategies,
        }

    @staticmethod
    def classify_retrieval(
        chunks: list[RetrievedChunk],
        gold_doc_ids: list[str] | None = None,
        min_relevance_score: float = 0.4,
    ) -> str:
        """Classify retrieval outcome for observability dashboards."""
        if not chunks:
            return "no_hit"
        top_score = chunks[0].score
        if top_score < min_relevance_score:
            return "low_score"

        if gold_doc_ids:
            gold_set = set(gold_doc_ids)
            doc_ids = [chunk.doc_id for chunk in chunks]
            if any(doc_id in gold_set for doc_id in doc_ids):
                if doc_ids[0] in gold_set:
                    return "hit_top1"
                return "hit_not_top1"
            return "wrong_document"

        return "retrieved_without_gold"

    def _expand_with_parent_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Append parent chunk text for child hits when parent metadata exists."""
        if not chunks:
            return chunks

        parent_ids = [chunk.metadata.get("parent_id") for chunk in chunks if chunk.metadata.get("parent_id")]
        if not parent_ids:
            return chunks

        parents = self.collection.get(ids=list(set(parent_ids)), include=["documents", "metadatas"])
        parent_doc_map: dict[str, str] = {}
        for idx, parent_id in enumerate(parents.get("ids", [])):
            docs = parents.get("documents", [])
            if idx < len(docs):
                parent_doc_map[parent_id] = docs[idx]

        enriched: list[RetrievedChunk] = []
        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id and parent_id in parent_doc_map:
                chunk.metadata["parent_text"] = parent_doc_map[parent_id]
            enriched.append(chunk)

        return enriched

    @staticmethod
    def _dedupe_results_by_doc(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep highest-ranked chunk per document for document-level evals."""
        seen_docs: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)
            deduped.append(chunk)
        return deduped
