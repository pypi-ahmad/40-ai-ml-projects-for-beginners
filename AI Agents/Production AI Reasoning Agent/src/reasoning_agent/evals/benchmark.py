"""Benchmark runner for multi-model evaluation."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from statistics import mean

from reasoning_agent.agent.models import AgentRequest
from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.evals.dataset import BenchmarkPrediction, BenchmarkPrompt
from reasoning_agent.evals.judge import LLMJudge


@dataclass(slots=True)
class BenchmarkSummary:
    model: str
    total_prompts: int
    success_rate: float
    avg_latency_ms: float
    avg_tool_calls: float
    avg_keyword_score: float
    avg_tool_selection_score: float
    avg_judge_score: float


@dataclass(slots=True)
class BenchmarkRunner:
    """Runs benchmark prompts across configured models."""

    runner: AgentRunner
    judge: LLMJudge | None = None

    async def run_for_model(self, model: str, prompts: list[BenchmarkPrompt]) -> list[BenchmarkPrediction]:
        predictions: list[BenchmarkPrediction] = []
        original_model = self.runner.settings.llm.primary_model
        self.runner.settings.llm.primary_model = model
        self.runner.workflow.planner.model = model
        self.runner.workflow.responder.model = model

        llm_needed = (
            self.runner.settings.agent.use_llm_for_planning
            or self.runner.settings.agent.use_llm_for_response
            or self.judge is not None
        )
        if llm_needed:
            available = await self.runner.llm.ensure_model_available(
                model=model,
                auto_pull=self.runner.settings.llm.auto_pull_missing_models,
            )
            if not available:
                for prompt in prompts:
                    predictions.append(
                        BenchmarkPrediction(
                            prompt_id=prompt.prompt_id,
                            category=prompt.category,
                            model=model,
                            answer="SKIPPED: model unavailable",
                            success=False,
                            latency_ms=0.0,
                            iterations=0,
                            tool_calls=0,
                            retry_count=0,
                            required_tools=prompt.required_tools,
                            used_tools=[],
                            keyword_score=0.0,
                            tool_selection_score=0.0,
                            judge_score=0.0,
                            judge_notes="model unavailable",
                        )
                    )
                self._restore_model(original_model)
                return predictions

        for prompt in prompts:
            started = time.perf_counter()
            result = await self.runner.run(
                AgentRequest(
                    query=prompt.prompt,
                    session_id=f"bench-{model.replace(':', '-')}",
                )
            )
            latency_ms = (time.perf_counter() - started) * 1000

            used_tools = [call.get("tool_name", "") for call in result.tool_calls]
            keyword_score = self._keyword_score(result.answer, prompt.expected_keywords)
            tool_score = self._tool_selection_score(used_tools, prompt.required_tools)

            prediction = BenchmarkPrediction(
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                model=model,
                answer=result.answer,
                success=result.success,
                latency_ms=latency_ms,
                iterations=int(result.metrics.get("iterations", 0)),
                tool_calls=len(result.tool_calls),
                retry_count=0,
                required_tools=prompt.required_tools,
                used_tools=used_tools,
                keyword_score=keyword_score,
                tool_selection_score=tool_score,
            )

            if self.judge is not None:
                judge_out = await self.judge.score(prompt, result.answer, used_tools)
                prediction.judge_score = (
                    judge_out.correctness
                    + judge_out.grounding
                    + judge_out.completeness
                    + judge_out.tool_usage
                ) / 4
                prediction.judge_notes = judge_out.notes

            predictions.append(prediction)

        self._restore_model(original_model)
        return predictions

    async def run_all(
        self,
        prompts: list[BenchmarkPrompt],
        models: list[str],
    ) -> tuple[list[BenchmarkPrediction], list[BenchmarkSummary]]:
        all_predictions: list[BenchmarkPrediction] = []
        summaries: list[BenchmarkSummary] = []

        for model in models:
            predictions = await self.run_for_model(model, prompts)
            all_predictions.extend(predictions)
            summaries.append(self._summarize(model, predictions))

        return all_predictions, summaries

    def _restore_model(self, model: str) -> None:
        self.runner.settings.llm.primary_model = model
        self.runner.workflow.planner.model = model
        self.runner.workflow.responder.model = model

    def _summarize(self, model: str, predictions: list[BenchmarkPrediction]) -> BenchmarkSummary:
        judge_values = [p.judge_score for p in predictions if p.judge_score is not None]
        return BenchmarkSummary(
            model=model,
            total_prompts=len(predictions),
            success_rate=mean([1.0 if p.success else 0.0 for p in predictions]) if predictions else 0.0,
            avg_latency_ms=mean([p.latency_ms for p in predictions]) if predictions else 0.0,
            avg_tool_calls=mean([float(p.tool_calls) for p in predictions]) if predictions else 0.0,
            avg_keyword_score=mean([p.keyword_score for p in predictions]) if predictions else 0.0,
            avg_tool_selection_score=mean([p.tool_selection_score for p in predictions]) if predictions else 0.0,
            avg_judge_score=mean(judge_values) if judge_values else 0.0,
        )

    def _keyword_score(self, answer: str, expected_keywords: list[str]) -> float:
        if not expected_keywords:
            return 1.0
        text = answer.lower()
        matched = sum(1 for keyword in expected_keywords if keyword.lower() in text)
        return matched / len(expected_keywords)

    def _tool_selection_score(self, used_tools: list[str], required_tools: list[str]) -> float:
        if not required_tools:
            return 1.0
        used = set(tool for tool in used_tools if tool)
        required = set(required_tools)
        if not used:
            return 0.0
        return len(used & required) / len(required)


async def run_benchmarks_async(
    runner: AgentRunner,
    judge: LLMJudge | None,
    prompts: list[BenchmarkPrompt],
    models: list[str],
) -> tuple[list[BenchmarkPrediction], list[BenchmarkSummary]]:
    bench = BenchmarkRunner(runner=runner, judge=judge)
    return await bench.run_all(prompts=prompts, models=models)


def run_benchmarks(
    runner: AgentRunner,
    judge: LLMJudge | None,
    prompts: list[BenchmarkPrompt],
    models: list[str],
) -> tuple[list[BenchmarkPrediction], list[BenchmarkSummary]]:
    return asyncio.run(run_benchmarks_async(runner=runner, judge=judge, prompts=prompts, models=models))
