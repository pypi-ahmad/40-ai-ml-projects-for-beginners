"""Retrieval and response evaluation utilities."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from local_rag.rag import RAGPipeline
from local_rag.retriever import Retriever
from local_rag.types import EvalExample, ResponseMetrics, RetrievalMetrics
from local_rag.utils import write_jsonl


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K."""

    top = retrieved_ids[:k]
    if not top:
        return 0.0
    seen_hits: set[str] = set()
    for item in top:
        if item in relevant_ids:
            seen_hits.add(item)
    hits = len(seen_hits)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K."""

    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    seen_hits: set[str] = set()
    for item in top:
        if item in relevant_ids:
            seen_hits.add(item)
    hits = len(seen_hits)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute reciprocal rank for first relevant hit."""

    for idx, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute NDCG@K with binary relevance."""

    dcg = 0.0
    seen_hits: set[str] = set()
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant_ids and item not in seen_hits:
            seen_hits.add(item)
            dcg += 1.0 / math.log2(idx + 1)

    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """Evaluate retriever quality against labeled examples."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def evaluate(
        self,
        examples: Iterable[EvalExample],
        ks: tuple[int, ...] = (3, 5, 10),
    ) -> list[RetrievalMetrics]:
        """Compute aggregate metrics for candidate K values."""

        rows = list(examples)
        if not rows:
            return []

        metrics: list[RetrievalMetrics] = []
        for k in ks:
            p_vals: list[float] = []
            r_vals: list[float] = []
            rr_vals: list[float] = []
            ndcg_vals: list[float] = []
            latencies: list[float] = []

            for row in rows:
                retrieved, retrieval_ms = self.retriever.retrieve(row.query, top_k=k)
                latencies.append(retrieval_ms)
                ids, relevant = self._ids_for_eval(row=row, retrieved=retrieved)

                p_vals.append(precision_at_k(ids, relevant, k))
                r_vals.append(recall_at_k(ids, relevant, k))
                rr_vals.append(reciprocal_rank(ids, relevant))
                ndcg_vals.append(ndcg_at_k(ids, relevant, k))

            metrics.append(
                RetrievalMetrics(
                    k=k,
                    precision_at_k=statistics.mean(p_vals),
                    recall_at_k=statistics.mean(r_vals),
                    mrr=statistics.mean(rr_vals),
                    ndcg=statistics.mean(ndcg_vals),
                    avg_retrieval_latency_ms=statistics.mean(latencies),
                )
            )

        return metrics

    @staticmethod
    def _ids_for_eval(
        *,
        row: EvalExample,
        retrieved,
    ) -> tuple[list[str], set[str]]:
        if row.relevant_chunk_ids:
            return [hit.chunk_id for hit in retrieved], set(row.relevant_chunk_ids)
        return [hit.doc_id for hit in retrieved], set(row.relevant_doc_ids)


def dump_retrieval_metrics(path: Path, metrics: list[RetrievalMetrics]) -> None:
    """Persist retrieval metrics to JSONL."""

    write_jsonl(path, [asdict(metric) for metric in metrics])


class ResponseEvaluator:
    """Evaluate generation latency/length/citations over query set."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(self, queries: Iterable[str], *, top_k: int) -> ResponseMetrics:
        """Compute response-level metrics."""

        rows = [query.strip() for query in queries if query.strip()]
        if not rows:
            return ResponseMetrics(
                avg_generation_latency_ms=0.0,
                avg_answer_length=0.0,
                avg_citation_count=0.0,
            )

        generation_latencies: list[float] = []
        answer_lengths: list[float] = []
        citation_counts: list[float] = []

        for query in rows:
            response = self.pipeline.ask(query, top_k=top_k)
            generation_latencies.append(response.timings.generation_ms)
            answer_lengths.append(float(len(response.answer.split())))
            citation_counts.append(float(len(response.citations)))

        return ResponseMetrics(
            avg_generation_latency_ms=statistics.mean(generation_latencies),
            avg_answer_length=statistics.mean(answer_lengths),
            avg_citation_count=statistics.mean(citation_counts),
        )
