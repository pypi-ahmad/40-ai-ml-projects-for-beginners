"""Offline evaluation and LLM-assisted judging."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable

import requests

from semantic_search.schemas import EvaluationCase, MetricRow, SearchResponse


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@k metric."""
    if k == 0:
        return 0.0
    top_k = _unique_preserve_order(retrieved)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@k metric."""
    if not relevant:
        return 0.0
    top_k = _unique_preserve_order(retrieved)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """MRR for one query."""
    for idx, doc_id in enumerate(_unique_preserve_order(retrieved), start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@k for binary relevance."""
    deduped = _unique_preserve_order(retrieved)
    dcg = 0.0
    for idx, doc_id in enumerate(deduped[:k], start=1):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(idx + 1)

    ideal_rel_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


@dataclass(slots=True)
class EvaluationOutput:
    summary: MetricRow
    per_query: list[dict[str, float | str]]


def evaluate_retrieval(
    system_name: str,
    cases: list[EvaluationCase],
    search_fn: Callable[[str], SearchResponse],
    k: int = 10,
) -> EvaluationOutput:
    """Evaluate retrieval function over labeled queries."""
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    mrr_vals: list[float] = []
    ndcg_vals: list[float] = []
    latencies: list[float] = []
    per_query: list[dict[str, float | str]] = []

    for case in cases:
        start = perf_counter()
        response = search_fn(case.query)
        latency_ms = (perf_counter() - start) * 1000

        retrieved_doc_ids = [hit.document_id for hit in response.hits]
        relevant = set(case.relevant_doc_ids)

        p = precision_at_k(retrieved_doc_ids, relevant, k)
        r = recall_at_k(retrieved_doc_ids, relevant, k)
        m = mean_reciprocal_rank(retrieved_doc_ids, relevant)
        n = ndcg_at_k(retrieved_doc_ids, relevant, k)

        precision_vals.append(p)
        recall_vals.append(r)
        mrr_vals.append(m)
        ndcg_vals.append(n)
        latencies.append(latency_ms)

        per_query.append(
            {
                "query_id": case.query_id,
                "precision": p,
                "recall": r,
                "mrr": m,
                "ndcg": n,
                "latency_ms": latency_ms,
            }
        )

    summary = MetricRow(
        system_name=system_name,
        precision_at_k=mean(precision_vals) if precision_vals else 0.0,
        recall_at_k=mean(recall_vals) if recall_vals else 0.0,
        mrr=mean(mrr_vals) if mrr_vals else 0.0,
        ndcg=mean(ndcg_vals) if ndcg_vals else 0.0,
        avg_latency_ms=mean(latencies) if latencies else 0.0,
    )
    return EvaluationOutput(summary=summary, per_query=per_query)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """Load evaluation set from JSONL."""
    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            cases.append(EvaluationCase.model_validate_json(line))
    return cases


def save_evaluation_output(output: EvaluationOutput, path: str | Path) -> None:
    """Save evaluation summary and per-query scores."""
    payload = {
        "summary": output.summary.model_dump(),
        "per_query": output.per_query,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class OllamaJudge:
    """LLM-assisted relevance and usefulness scoring."""

    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def judge(self, query: str, results: list[str]) -> dict[str, float | str]:
        """Return structured judgement from Ollama model."""
        prompt = (
            "Score search results from 1-5 for relevance, grounding, and coverage. "
            "Return strict JSON with keys relevance, grounding, coverage, notes.\n"
            f"Query: {query}\n"
            f"Results: {results[:5]}"
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return json.loads(data.get("response", "{}"))
        except json.JSONDecodeError:
            return {"relevance": 0, "grounding": 0, "coverage": 0, "notes": "invalid_json"}
