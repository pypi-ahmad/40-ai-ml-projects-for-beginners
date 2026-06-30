"""Benchmark runner across local models."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.evaluation.datasets import BenchmarkDataset
from reasoning_agent.evaluation.judge import JudgeInput, JudgeRunner
from reasoning_agent.llm.ollama_client import OllamaClient
from reasoning_agent.schemas import ReasoningMode
from reasoning_agent.settings import Settings
from reasoning_agent.utils.json_utils import dumps


@dataclass(slots=True)
class BenchmarkSummary:
    """Per-model benchmark summary."""

    model: str
    available: bool
    prompts: int
    completed: int
    accuracy: float
    avg_latency_ms: float
    avg_tool_calls: float
    avg_retries: float
    success_rate: float
    skip_reason: str = ""


class BenchmarkRunner:
    """Run full benchmark suite across configured models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Execute full benchmark run and persist reports."""

        dataset = BenchmarkDataset.load_jsonl(self.settings.benchmark_dataset_path)
        target_dir = Path(output_dir or self.settings.benchmark_output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        llm = OllamaClient(self.settings.ollama_base_url, self.settings.request_timeout_seconds)
        available = set(llm.available_models())

        models = [self.settings.primary_model, *self.settings.compare_model_list]
        judge = JudgeRunner(llm, judge_model="granite4.1:3b")

        summaries: list[BenchmarkSummary] = []
        details: dict[str, Any] = {}

        for model in models:
            if model not in available:
                summaries.append(
                    BenchmarkSummary(
                        model=model,
                        available=False,
                        prompts=len(dataset.records),
                        completed=0,
                        accuracy=0.0,
                        avg_latency_ms=0.0,
                        avg_tool_calls=0.0,
                        avg_retries=0.0,
                        success_rate=0.0,
                        skip_reason="model_not_available",
                    )
                )
                details[model] = {"skipped": True, "reason": "model_not_available"}
                continue

            tuned = self.settings.model_copy(update={"primary_model": model})
            runner = AgentRunner(settings=tuned)

            model_rows: list[dict[str, Any]] = []
            latencies: list[float] = []
            tool_calls: list[int] = []
            retries: list[float] = []
            scores: list[float] = []
            successes = 0

            for record in dataset.records:
                started = time.perf_counter()
                response = runner.run(session_id=f"bench-{model}", user_input=record.prompt, mode=ReasoningMode.REACT)
                latency = (time.perf_counter() - started) * 1000

                latencies.append(latency)
                tool_count = len([t for t in response.trace if t.action is not None])
                tool_calls.append(tool_count)
                retries.append(float(response.metrics.get("retries", 0)))

                if response.success:
                    successes += 1

                judge_row = judge.score(
                    JudgeInput(
                        prompt=record.prompt,
                        answer=response.answer,
                        expected_keywords=record.expected_keywords,
                    )
                )
                scores.append(float(judge_row["hybrid"]))

                model_rows.append(
                    {
                        "prompt_id": record.prompt_id,
                        "category": record.category,
                        "prompt": record.prompt,
                        "answer": response.answer,
                        "success": response.success,
                        "termination_reason": response.termination_reason,
                        "latency_ms": latency,
                        "tool_calls": tool_count,
                        "retries": response.metrics.get("retries", 0),
                        "judge": judge_row,
                    }
                )

            completed = len(model_rows)
            summary = BenchmarkSummary(
                model=model,
                available=True,
                prompts=len(dataset.records),
                completed=completed,
                accuracy=(sum(scores) / completed) if completed else 0.0,
                avg_latency_ms=(sum(latencies) / completed) if completed else 0.0,
                avg_tool_calls=(sum(tool_calls) / completed) if completed else 0.0,
                avg_retries=(sum(retries) / completed) if completed else 0.0,
                success_rate=(successes / completed) if completed else 0.0,
            )
            summaries.append(summary)
            details[model] = {"skipped": False, "rows": model_rows}

            runner.close()

        llm.close()

        summary_payload = [asdict(s) for s in summaries]
        (target_dir / "benchmark_summary.json").write_text(dumps(summary_payload, indent=True), encoding="utf-8")
        (target_dir / "benchmark_details.json").write_text(dumps(details, indent=True), encoding="utf-8")

        return {"summary": summary_payload, "details": details}
