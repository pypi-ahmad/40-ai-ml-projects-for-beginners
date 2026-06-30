"""Retrieval algorithms: semantic, lexical, hybrid, and MMR."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from semantic_search.schemas import DocumentChunk, SearchHit


def _tokenize(text: str) -> list[str]:
    return [token.strip() for token in text.lower().split() if token.strip()]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Compute RRF score across rank lists."""
    scores: dict[str, float] = defaultdict(float)
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    vector_latency_ms: float
    lexical_latency_ms: float


class HybridRetriever:
    """Unified retrieval engine across semantic and lexical indexes."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        vector_metric: str = "cosine",
        mmr_lambda: float = 0.6,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.vector_metric = vector_metric
        self.mmr_lambda = mmr_lambda
        self.rrf_k = rrf_k

        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.corpus_tokens = [_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self._tfidf_vectorizer = TfidfVectorizer(min_df=1)
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform([chunk.text for chunk in chunks])

    def lexical_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """BM25 lexical search."""
        start = perf_counter()
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_indices = np.argsort(scores)[::-1]

        hits: list[SearchHit] = []
        for idx in ranked_indices:
            chunk = self.chunks[int(idx)]
            if not self._passes_filters(chunk, filters):
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(scores[int(idx)]),
                    retrieval_source="lexical",
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break

        latency_ms = (perf_counter() - start) * 1000
        return hits, latency_ms

    def semantic_search(
        self,
        vector_response: dict[str, list],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Convert vector store response into SearchHit list."""
        start = perf_counter()
        docs = vector_response.get("documents", [[]])[0]
        metadatas = vector_response.get("metadatas", [[]])[0]
        distances = vector_response.get("distances", [[]])[0]

        hits: list[SearchHit] = []
        for idx, text in enumerate(docs):
            metadata = dict(metadatas[idx]) if idx < len(metadatas) else {}
            chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or metadata.get("chunkId") or "")
            if not chunk_id and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_by_id.get(chunk_id)
            if chunk and not self._passes_filters(chunk, filters):
                continue

            raw_distance = float(distances[idx]) if idx < len(distances) else 0.0
            if self.vector_metric == "cosine":
                score = 1.0 - raw_distance
            elif self.vector_metric == "euclidean":
                score = 1.0 / (1.0 + raw_distance)
            else:
                score = raw_distance

            resolved_chunk = chunk or DocumentChunk(
                chunk_id=chunk_id or f"chroma-{idx}",
                document_id=str(metadata.get("document_id") or metadata.get("documentId") or "unknown"),
                chunk_index=int(metadata.get("chunk_index") or idx),
                text=text,
                metadata=metadata,
            )

            hits.append(
                SearchHit(
                    chunk_id=resolved_chunk.chunk_id,
                    document_id=resolved_chunk.document_id,
                    text=resolved_chunk.text,
                    metadata=resolved_chunk.metadata,
                    score=score,
                    retrieval_source="semantic",
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break

        latency_ms = (perf_counter() - start) * 1000
        return hits, latency_ms

    def hybrid_search(
        self,
        semantic_hits: list[SearchHit],
        lexical_hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        """Fuse semantic and lexical ranked lists using RRF."""
        semantic_ranked = [hit.chunk_id for hit in semantic_hits]
        lexical_ranked = [hit.chunk_id for hit in lexical_hits]
        rrf_scores = reciprocal_rank_fusion([semantic_ranked, lexical_ranked], k=self.rrf_k)

        merged_hits: dict[str, SearchHit] = {}
        for hit in semantic_hits + lexical_hits:
            if hit.chunk_id not in merged_hits:
                merged_hits[hit.chunk_id] = hit

        ranked = sorted(merged_hits.values(), key=lambda h: rrf_scores.get(h.chunk_id, 0.0), reverse=True)
        output: list[SearchHit] = []
        for idx, hit in enumerate(ranked[:top_k]):
            output.append(
                hit.model_copy(
                    update={
                        "score": float(rrf_scores.get(hit.chunk_id, hit.score)),
                        "retrieval_source": "hybrid",
                        "rank": idx + 1,
                    }
                )
            )
        return output

    def apply_mmr(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        """Apply MMR selection to diversify ranked hits."""
        if len(hits) <= 1:
            return hits

        texts = [hit.text for hit in hits]
        all_vectors = self._tfidf_vectorizer.transform(texts)
        query_vector = self._tfidf_vectorizer.transform([query])

        selected: list[int] = []
        candidates = set(range(len(hits)))

        similarity_to_query = (all_vectors @ query_vector.T).toarray().flatten()
        first_idx = int(np.argmax(similarity_to_query))
        selected.append(first_idx)
        candidates.remove(first_idx)

        while candidates and len(selected) < top_k:
            best_score = -math.inf
            best_idx = None
            for candidate in candidates:
                sim_query = float(similarity_to_query[candidate])
                diversity_penalty = 0.0
                for chosen in selected:
                    pairwise = float((all_vectors[candidate] @ all_vectors[chosen].T).toarray()[0, 0])
                    diversity_penalty = max(diversity_penalty, pairwise)
                mmr_score = self.mmr_lambda * sim_query - (1 - self.mmr_lambda) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = candidate
            if best_idx is None:
                break
            selected.append(best_idx)
            candidates.remove(best_idx)

        mmr_hits = [hits[idx] for idx in selected]
        return [hit.model_copy(update={"rank": rank + 1, "retrieval_source": "mmr"}) for rank, hit in enumerate(mmr_hits)]

    @staticmethod
    def _passes_filters(chunk: DocumentChunk, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if key == "date_from":
                actual_date = str(chunk.metadata.get("date") or "")
                if actual_date and str(actual_date) < str(expected):
                    return False
                continue
            if key == "date_to":
                actual_date = str(chunk.metadata.get("date") or "")
                if actual_date and str(actual_date) > str(expected):
                    return False
                continue

            actual = chunk.metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True
