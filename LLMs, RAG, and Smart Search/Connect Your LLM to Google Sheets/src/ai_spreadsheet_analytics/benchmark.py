"""Benchmark suite for analytics Q&A and model comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Callable
from uuid import uuid4

from ai_spreadsheet_analytics.schemas import BenchmarkCase, BenchmarkResult
from ai_spreadsheet_analytics.state_store import SQLiteStateStore


class BenchmarkRunner:
    """Run benchmark cases and persist metrics."""

    def __init__(self, state_store: SQLiteStateStore) -> None:
        self.state_store = state_store

    def load_cases(self, path: Path) -> list[BenchmarkCase]:
        """Load benchmark cases from JSON."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [BenchmarkCase(**item) for item in payload]

    def run(
        self,
        model: str,
        cases: list[BenchmarkCase],
        answer_fn: Callable[[str], tuple[str, float]],
    ) -> tuple[str, list[BenchmarkResult], dict[str, float]]:
        """Run benchmark evaluation for one model.

        Args:
            model: Model identifier.
            cases: Benchmark cases.
            answer_fn: Function returning (answer, latency_ms) for question.

        Returns:
            Run ID, per-case results, aggregate metrics.
        """
        run_id = f"bench_{uuid4().hex[:10]}"
        results: list[BenchmarkResult] = []

        for case in cases:
            answer, latency_ms = answer_fn(case.question)
            lower = answer.lower()
            passed_keywords = sum(1 for kw in case.expected_keywords if kw.lower() in lower)
            total_keywords = max(len(case.expected_keywords), 1)
            consistency = passed_keywords / total_keywords
            hallucination_flag = case.expected_type == "numeric" and any(
                token in lower for token in ["i think", "maybe", "probably"]
            )
            usefulness = min(1.0, 0.4 + consistency)

            result = BenchmarkResult(
                run_id=run_id,
                case_id=case.case_id,
                model=model,
                latency_ms=latency_ms,
                passed_keywords=passed_keywords,
                total_keywords=total_keywords,
                hallucination_flag=hallucination_flag,
                consistency_score=consistency,
                usefulness_score=usefulness,
            )
            results.append(result)
            self.state_store.add_benchmark_result(run_id, case.case_id, model, asdict(result))

        aggregate = {
            "cases": float(len(results)),
            "avg_latency_ms": mean(r.latency_ms for r in results) if results else 0.0,
            "keyword_accuracy": mean(r.passed_keywords / r.total_keywords for r in results) if results else 0.0,
            "hallucination_rate": mean(1.0 if r.hallucination_flag else 0.0 for r in results) if results else 0.0,
            "consistency": mean(r.consistency_score for r in results) if results else 0.0,
            "business_usefulness": mean(r.usefulness_score for r in results) if results else 0.0,
        }
        return run_id, results, aggregate

    def save_results(self, output_path: Path, run_id: str, results: list[BenchmarkResult], aggregate: dict[str, float]) -> Path:
        """Save benchmark outputs to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "aggregate": aggregate,
            "results": [asdict(result) for result in results],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
