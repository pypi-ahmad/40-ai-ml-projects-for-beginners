"""Metrics for retrieval and generation evaluation in RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalMetricRow:
    """Single-query retrieval metric row."""

    query_id: str
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    reciprocal_rank: float
    ndcg: float


@dataclass(slots=True)
class RetrievalMetricSummary:
    """Aggregated retrieval metric summary."""

    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg: float
    num_queries: int


@dataclass(slots=True)
class GenerationMetricSummary:
    """Aggregated generation metric summary."""

    exact_match: float
    bleu: float
    rouge1: float
    rougeL: float
    meteor: float
    bertscore_f1: float
    num_examples: int


@lru_cache(maxsize=1)
def _load_metric(name: str):
    """Lazy-load evaluate metric object once per process."""
    import evaluate

    return evaluate.load(name)


def compute_retrieval_metrics(
    query_ids: Sequence[str],
    retrieved_doc_ids: Sequence[Sequence[str]],
    gold_doc_ids: Sequence[Sequence[str]],
    top_k: int,
) -> tuple[RetrievalMetricSummary, list[RetrievalMetricRow]]:
    """Compute retrieval metrics for each query and aggregate summary."""
    rows: list[RetrievalMetricRow] = []

    for qid, retrieved, gold in zip(query_ids, retrieved_doc_ids, gold_doc_ids):
        retrieved_unique = _dedupe_keep_order(list(retrieved))
        retrieved_k = retrieved_unique[:top_k]
        gold_set = set(gold)

        if not gold_set:
            continue

        hits = [1 if doc_id in gold_set else 0 for doc_id in retrieved_k]
        tp = sum(hits)

        precision = tp / len(retrieved_k) if retrieved_k else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rr = 0.0
        for idx, hit in enumerate(hits, start=1):
            if hit:
                rr = 1.0 / idx
                break

        ndcg_val = _binary_ndcg_at_k(hits=hits, total_relevant=len(gold_set), top_k=top_k)

        rows.append(
            RetrievalMetricRow(
                query_id=qid,
                precision_at_k=precision,
                recall_at_k=recall,
                f1_at_k=f1,
                reciprocal_rank=rr,
                ndcg=ndcg_val,
            )
        )

    if not rows:
        empty = RetrievalMetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return empty, []

    summary = RetrievalMetricSummary(
        precision_at_k=float(np.mean([row.precision_at_k for row in rows])),
        recall_at_k=float(np.mean([row.recall_at_k for row in rows])),
        f1_at_k=float(np.mean([row.f1_at_k for row in rows])),
        mrr=float(np.mean([row.reciprocal_rank for row in rows])),
        ndcg=float(np.mean([row.ndcg for row in rows])),
        num_queries=len(rows),
    )
    return summary, rows


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _binary_ndcg_at_k(hits: Sequence[int], total_relevant: int, top_k: int) -> float:
    """Compute NDCG@K for binary relevance."""
    if top_k <= 0:
        return 0.0
    if not hits:
        return 0.0

    truncated = list(hits[:top_k])
    dcg = 0.0
    for rank, rel in enumerate(truncated, start=1):
        if rel > 0:
            dcg += 1.0 / np.log2(rank + 1.0)

    ideal_relevant = min(total_relevant, top_k)
    if ideal_relevant == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(rank + 1.0) for rank in range(1, ideal_relevant + 1))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_generation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    bertscore_model: str = "distilbert-base-uncased",
) -> GenerationMetricSummary:
    """Compute generation metrics for RAG outputs.

    Metrics include exact match and common NLG metrics used in RAG papers.
    """
    pairs = [(p.strip(), r.strip()) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return GenerationMetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    preds = [p for p, _ in pairs]
    refs = [r for _, r in pairs]

    em = float(np.mean([1.0 if p.lower() == r.lower() else 0.0 for p, r in pairs]))

    bleu_metric = _load_metric("bleu")
    bleu_refs = [[reference] for reference in refs]
    bleu = float(bleu_metric.compute(predictions=preds, references=bleu_refs).get("bleu", 0.0))

    rouge_metric = _load_metric("rouge")
    rouge = rouge_metric.compute(predictions=preds, references=refs)
    rouge1 = float(rouge.get("rouge1", 0.0))
    rouge_l = float(rouge.get("rougeL", 0.0))

    meteor_metric = _load_metric("meteor")
    meteor = float(meteor_metric.compute(predictions=preds, references=refs).get("meteor", 0.0))

    bertscore_metric = _load_metric("bertscore")
    bertscore = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        model_type=bertscore_model,
        lang="en",
    )
    f1_values = bertscore.get("f1", [])
    bertscore_f1 = float(np.mean(f1_values)) if f1_values else 0.0

    return GenerationMetricSummary(
        exact_match=em,
        bleu=bleu,
        rouge1=rouge1,
        rougeL=rouge_l,
        meteor=meteor,
        bertscore_f1=bertscore_f1,
        num_examples=len(preds),
    )


def summarize_retrieval_rows(rows: Iterable[RetrievalMetricRow]) -> dict[str, float]:
    """Convert retrieval rows into compact mean summary dict."""
    rows_list = list(rows)
    if not rows_list:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
        }

    return {
        "precision_at_k": float(np.mean([row.precision_at_k for row in rows_list])),
        "recall_at_k": float(np.mean([row.recall_at_k for row in rows_list])),
        "f1_at_k": float(np.mean([row.f1_at_k for row in rows_list])),
        "mrr": float(np.mean([row.reciprocal_rank for row in rows_list])),
        "ndcg": float(np.mean([row.ndcg for row in rows_list])),
    }
