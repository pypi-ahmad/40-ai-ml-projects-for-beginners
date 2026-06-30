#!/usr/bin/env python3
"""Run full benchmark matrix and save artifacts."""

from __future__ import annotations

from pathlib import Path

from ai_sql_assistant.benchmarking.benchmark import BenchmarkRunner
from ai_sql_assistant.config import get_settings
from ai_sql_assistant.logging_utils import configure_logging, logger
from ai_sql_assistant.pipeline.assistant import AISQLAssistant


def main() -> None:
    configure_logging()
    settings = get_settings()
    assistant = AISQLAssistant(settings)
    runner = BenchmarkRunner(assistant=assistant, settings=settings)

    try:
        cases = runner.load_cases(Path("benchmarks/benchmark_cases.json"))
        run = runner.run(cases)
        case_map = {case.case_id: case for case in cases}
        judge = runner.evaluate_with_judge(case_map, run)
        out_path = runner.save_run(run, judge)
        logger.info("Benchmark saved to {}", out_path)
        logger.info("Summary: {}", run.metrics)
    finally:
        runner.close()
        assistant.close()


if __name__ == "__main__":
    main()
