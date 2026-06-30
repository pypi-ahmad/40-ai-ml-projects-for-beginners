"""Run full benchmark suite and emit reports."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import polars as pl

from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.config import get_settings
from reasoning_agent.constants import ARTIFACTS_DIR, BENCHMARK_DATA_PATH
from reasoning_agent.evals import LLMJudge, load_benchmark_prompts, run_benchmarks, save_benchmark_reports
from reasoning_agent.llm.ollama import OllamaProvider
from reasoning_agent.observability.visualization import (
    latency_bar,
    radar_quality,
    success_rate_bar,
    tool_usage_chart,
)


def main() -> None:
    settings = get_settings()
    settings.memory.chroma_enabled = False
    settings.agent.runtime_mode = "fallback"
    settings.agent.graph_timeout_seconds = 1
    llm = OllamaProvider(settings.llm.base_url, timeout_seconds=settings.llm.request_timeout_seconds)
    llm_available = asyncio.run(llm.healthcheck())
    if not llm_available:
        os.environ["AGENT_OFFLINE_MODE"] = "1"
        settings.agent.use_llm_for_planning = False
        settings.agent.use_llm_for_response = False

    runner = AgentRunner(settings=settings)

    prompts = load_benchmark_prompts(Path(BENCHMARK_DATA_PATH))
    judge = LLMJudge(llm=llm, model=settings.llm.llm_judge_model) if llm_available else None

    predictions, summaries = run_benchmarks(
        runner=runner,
        judge=judge,
        prompts=prompts,
        models=settings.llm.benchmark_models,
    )

    artifacts_dir = Path(ARTIFACTS_DIR) / "benchmarks"
    files = save_benchmark_reports(artifacts_dir, predictions, summaries)

    summary_df = pl.read_csv(files["summary_csv"])
    latency_bar(summary_df).write_html(artifacts_dir / "latency.html")
    success_rate_bar(summary_df).write_html(artifacts_dir / "success_rate.html")
    tool_usage_chart(summary_df).write_html(artifacts_dir / "tool_usage.html")
    radar_quality(summary_df).write_html(artifacts_dir / "quality_radar.html")

    print("Benchmark complete")
    for name, path in files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
