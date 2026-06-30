"""Evaluation framework for benchmark and retrieval/generation quality."""

from __future__ import annotations

import statistics
from pathlib import Path

from hybrid_research_assistant.schemas import EvalReport, EvalResult, EvalSample, RetrievalMode
from hybrid_research_assistant.utils import read_jsonl, write_jsonl
from hybrid_research_assistant.workflow import WorkflowRuntime


def load_eval_samples(path: Path) -> list[EvalSample]:
    """Load benchmark dataset rows."""

    rows = read_jsonl(path)
    return [EvalSample(**row) for row in rows]


def keyword_overlap_score(answer: str, keywords: list[str]) -> float:
    """Simple answer relevance approximation using keyword overlap."""

    if not keywords:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def citation_precision(answer_sources: list[str], expected_sources: list[str]) -> float:
    """Context precision approximation from source overlap."""

    if not answer_sources:
        return 0.0
    if not expected_sources:
        return 0.0
    expected = set(expected_sources)
    observed = set(answer_sources)
    tp = len(expected.intersection(observed))
    return tp / max(1, len(observed))


def evaluate_benchmark(
    runtime: WorkflowRuntime,
    samples: list[EvalSample],
    *,
    force_mode: RetrievalMode | None = None,
) -> tuple[list[EvalResult], EvalReport]:
    """Run benchmark set and compute aggregate metrics."""

    rows: list[EvalResult] = []
    for sample in samples:
        response = runtime.ask(sample.question, mode=force_mode or RetrievalMode.AUTO)
        source_files = [citation.source_file for citation in response.citations]
        relevance = keyword_overlap_score(response.answer, sample.expected_keywords)
        precision = citation_precision(source_files, sample.expected_sources)

        mode_correct = 1.0 if response.mode == sample.expected_mode else 0.0
        judge = response.judge
        faithfulness = float(judge.get("grounding", 1)) / 5.0 if judge else 0.0
        accuracy = (mode_correct + relevance) / 2.0

        rows.append(
            EvalResult(
                sample_id=sample.id,
                question=sample.question,
                category=sample.category,
                mode_used=response.mode,
                accuracy=accuracy,
                faithfulness=faithfulness,
                relevance=relevance,
                context_precision=precision,
                latency_ms=response.timings.total_ms,
                citations_count=len(response.citations),
            )
        )

    report = EvalReport(
        total_samples=len(rows),
        accuracy=statistics.mean(row.accuracy for row in rows) if rows else 0.0,
        faithfulness=statistics.mean(row.faithfulness for row in rows) if rows else 0.0,
        relevance=statistics.mean(row.relevance for row in rows) if rows else 0.0,
        context_precision=statistics.mean(row.context_precision for row in rows) if rows else 0.0,
        mean_latency_ms=statistics.mean(row.latency_ms for row in rows) if rows else 0.0,
        precision_at_k=statistics.mean(row.context_precision for row in rows) if rows else 0.0,
        recall_at_k=statistics.mean(row.relevance for row in rows) if rows else 0.0,
    )
    return rows, report


def save_eval_results(path: Path, rows: list[EvalResult]) -> None:
    """Persist evaluation result rows."""

    write_jsonl(path, [row.model_dump() for row in rows])
