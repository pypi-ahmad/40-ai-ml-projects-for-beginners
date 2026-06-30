"""Benchmarking pipeline for SQL generation quality and latency."""

from __future__ import annotations

import json
import resource
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ai_sql_assistant.benchmarking.judge import SQLJudge
from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.constants import REPORTS_DIR
from ai_sql_assistant.execution.executor import QueryExecutor
from ai_sql_assistant.logging_utils import logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant
from ai_sql_assistant.types import BenchmarkCase, BenchmarkCaseResult, BenchmarkRun, QueryRequest
from ai_sql_assistant.utils.sql_utils import normalize_sql


class BenchmarkRunner:
    """Run benchmark matrix across models and generation approaches."""

    def __init__(self, assistant: AISQLAssistant, settings: AppSettings) -> None:
        self.assistant = assistant
        self.settings = settings
        self.executor = QueryExecutor(settings.database.active_db_path)
        self.judge = SQLJudge(settings)

    def close(self) -> None:
        """Close shared resources."""
        self.judge.close()

    @staticmethod
    def load_cases(path: Path) -> list[BenchmarkCase]:
        """Load benchmark cases from JSON file."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [BenchmarkCase(**item) for item in payload]

    def run(self, cases: list[BenchmarkCase]) -> BenchmarkRun:
        """Run full benchmark matrix and return aggregate run object."""
        run = BenchmarkRun(run_id=str(uuid.uuid4()))

        models = [self.settings.models.generator_model, self.settings.models.comparison_model]
        approaches = ["langchain", "direct"]

        for case in cases:
            gt_exec = self.executor.execute(case.ground_truth_sql)
            gt_df = pd.DataFrame(gt_exec.rows)

            for approach in approaches:
                for model in models:
                    logger.info("Benchmark case={} approach={} model={}", case.case_id, approach, model)
                    request = QueryRequest(
                        question=case.question,
                        persona="Business Analyst",
                        user_id="benchmark",
                        conversation_id=f"bench-{case.case_id}",
                    )
                    try:
                        response = self.assistant.ask(
                            request=request,
                            approach=approach,
                            model=model,
                            generate_explanation=False,
                            enable_recovery=False,
                        )
                    except Exception as exc:
                        run.results.append(
                            BenchmarkCaseResult(
                                case_id=case.case_id,
                                model=model,
                                approach=approach,
                                generated_sql="",
                                exact_match=False,
                                execution_accuracy=False,
                                result_correctness=False,
                                generation_latency_ms=0.0,
                                execution_latency_ms=0.0,
                                complexity_score=0.0,
                                row_count=0,
                                token_count_estimate=0,
                                memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                                error=f"assistant.ask failed: {exc}",
                            )
                        )
                        logger.exception(
                            "Benchmark case failed case={} approach={} model={}",
                            case.case_id,
                            approach,
                            model,
                        )
                        continue

                    gen_sql_norm = normalize_sql(response.execution.sql)
                    gt_sql_norm = normalize_sql(case.ground_truth_sql)

                    execution_ok = response.execution.status == "success"
                    pred_df = pd.DataFrame(response.execution.rows)
                    result_correctness = self._result_equivalent(gt_df, pred_df)

                    result = BenchmarkCaseResult(
                        case_id=case.case_id,
                        model=model,
                        approach=approach,
                        generated_sql=response.execution.sql,
                        exact_match=gen_sql_norm == gt_sql_norm,
                        execution_accuracy=execution_ok,
                        result_correctness=result_correctness,
                        generation_latency_ms=response.generation.latency_ms,
                        execution_latency_ms=response.execution.execution_time_ms,
                        complexity_score=response.execution.complexity_score,
                        row_count=response.execution.row_count,
                        token_count_estimate=len(response.execution.sql.split()),
                        memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                        error=response.execution.error_message,
                    )
                    run.results.append(result)

        run.metrics = self._aggregate_metrics(run)
        return run

    def evaluate_with_judge(self, cases: dict[str, BenchmarkCase], run: BenchmarkRun) -> dict[str, Any]:
        """Score benchmark outputs with LLM-as-a-judge."""
        judge_rows: list[dict[str, Any]] = []
        for result in run.results:
            case = cases[result.case_id]
            try:
                score = self.judge.score(
                    question=case.question,
                    generated_sql=result.generated_sql,
                    ground_truth_sql=case.ground_truth_sql,
                )
                judge_rows.append(
                    {
                        "case_id": result.case_id,
                        "model": result.model,
                        "approach": result.approach,
                        **score.model_dump(),
                    }
                )
            except Exception as exc:
                judge_rows.append(
                    {
                        "case_id": result.case_id,
                        "model": result.model,
                        "approach": result.approach,
                        "sql_correctness": 0.0,
                        "business_correctness": 0.0,
                        "completeness": 0.0,
                        "readability": 0.0,
                        "efficiency": 0.0,
                        "safety": 0.0,
                        "rationale": f"Judge failed: {exc}",
                    }
                )

        judge_summary = pd.DataFrame(judge_rows).groupby(["model", "approach"]).mean(numeric_only=True).reset_index()
        return {
            "judge_rows": judge_rows,
            "judge_summary": judge_summary.to_dict(orient="records"),
        }

    @staticmethod
    def _result_equivalent(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
        if expected.empty and actual.empty:
            return True
        if set(expected.columns) != set(actual.columns):
            return False

        cols = sorted(expected.columns)
        exp = expected[cols].sort_values(cols).reset_index(drop=True)
        act = actual[cols].sort_values(cols).reset_index(drop=True)

        try:
            pd.testing.assert_frame_equal(exp, act, check_dtype=False, check_like=True, atol=1e-6, rtol=1e-6)
            return True
        except AssertionError:
            return False

    @staticmethod
    def _aggregate_metrics(run: BenchmarkRun) -> dict[str, Any]:
        frame = pd.DataFrame([item.model_dump() for item in run.results])
        if frame.empty:
            return {}

        summary = (
            frame.groupby(["model", "approach"]).agg(
                exact_match_rate=("exact_match", "mean"),
                execution_accuracy_rate=("execution_accuracy", "mean"),
                result_correctness_rate=("result_correctness", "mean"),
                avg_generation_latency_ms=("generation_latency_ms", "mean"),
                avg_execution_latency_ms=("execution_latency_ms", "mean"),
                avg_complexity_score=("complexity_score", "mean"),
                avg_rows_returned=("row_count", "mean"),
                avg_token_count=("token_count_estimate", "mean"),
                avg_memory_mb=("memory_mb", "mean"),
            )
        ).reset_index()

        summary["query_throughput_qps"] = 1000.0 / (
            summary["avg_generation_latency_ms"] + summary["avg_execution_latency_ms"]
        )

        return {
            "summary": summary.to_dict(orient="records"),
            "total_cases": int(frame["case_id"].nunique()),
            "total_runs": int(len(frame)),
        }

    @staticmethod
    def save_run(run: BenchmarkRun, judge_data: dict[str, Any] | None = None) -> Path:
        """Save benchmark run and optional judge report."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"benchmark_run_{timestamp}.json"

        payload: dict[str, Any] = run.model_dump()
        if judge_data:
            payload["judge"] = judge_data

        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
