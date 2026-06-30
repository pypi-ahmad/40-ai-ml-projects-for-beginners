"""Advanced retrieval features: query expansion, multi-query retrieval, reranking."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_system.generation import GenerationEngine
from rag_system.prompts import PromptLibrary
from rag_system.retrieval import RetrievalEngine
from rag_system.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdvancedRetrievalOutput:
    """Output payload for advanced retrieval experiments."""

    query: str
    expanded_queries: list[str]
    chunks: list[RetrievedChunk]


class AdvancedRetriever:
    """Adds query expansion and reranking on top of base retriever."""

    def __init__(self, retrieval_engine: RetrievalEngine, generation_engine: GenerationEngine) -> None:
        self.retrieval_engine = retrieval_engine
        self.generation_engine = generation_engine

    def expand_query(self, query: str, n: int = 4) -> list[str]:
        """Generate alternate retrieval queries using local LLM."""
        messages = PromptLibrary.query_expansion(query=query, n=n)
        response = self.generation_engine.generate(
            messages=messages,
            temperature=0.1,
            max_tokens=220,
            think=False,
            response_format={"type": "array", "items": {"type": "string"}},
        )

        text = response.get("text", "").strip()
        if not text:
            return [query]

        try:
            parsed = json.loads(text)
            expanded = [str(item).strip() for item in parsed if str(item).strip()]
            return [query, *expanded][: n + 1]
        except json.JSONDecodeError:
            # Fallback parser for plain-text list outputs.
            lines = [line.strip("-• \t") for line in text.splitlines()]
            expanded = [line for line in lines if line]
            return [query, *expanded][: n + 1]

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 6) -> list[RetrievedChunk]:
        """Rerank retrieved chunks using lexical + semantic hybrid score."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        doc_matrix = vectorizer.fit_transform(texts)
        query_vec = vectorizer.transform([query])
        lexical = cosine_similarity(query_vec, doc_matrix).flatten()

        for idx, chunk in enumerate(chunks):
            lexical_score = float(lexical[idx])
            semantic_score = chunk.score
            hybrid = 0.6 * semantic_score + 0.4 * lexical_score
            chunk.metadata["lexical_score"] = lexical_score
            chunk.metadata["hybrid_score"] = hybrid

        ranked = sorted(chunks, key=lambda c: float(c.metadata.get("hybrid_score", 0.0)), reverse=True)
        return ranked[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        expanded_queries: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> AdvancedRetrievalOutput:
        """Execute expansion -> multi-query retrieve -> rerank flow."""
        query_set = self.expand_query(query, n=expanded_queries)
        merged = self.retrieval_engine.multi_query(
            queries=query_set,
            top_k=max(top_k * 3, 12),
            metadata_filter=metadata_filter,
        )
        reranked = self.rerank(query=query, chunks=merged, top_k=top_k)

        logger.info(
            "Advanced retrieval produced %d chunks from %d query variants",
            len(reranked),
            len(query_set),
        )
        return AdvancedRetrievalOutput(query=query, expanded_queries=query_set, chunks=reranked)

    def compare_base_vs_advanced(self, query: str, top_k: int = 6) -> dict[str, object]:
        """Compare chunk ids/scores between base retrieval and advanced flow."""
        base = self.retrieval_engine.query(query=query, top_k=top_k)
        advanced = self.retrieve(query=query, top_k=top_k)

        base_ids = [chunk.chunk_id for chunk in base]
        adv_ids = [chunk.chunk_id for chunk in advanced.chunks]
        overlap = len(set(base_ids) & set(adv_ids))

        return {
            "query": query,
            "base_ids": base_ids,
            "advanced_ids": adv_ids,
            "overlap": overlap,
            "base_mean_score": float(np.mean([chunk.score for chunk in base])) if base else 0.0,
            "advanced_mean_hybrid": float(
                np.mean([float(chunk.metadata.get("hybrid_score", 0.0)) for chunk in advanced.chunks])
            )
            if advanced.chunks
            else 0.0,
        }
