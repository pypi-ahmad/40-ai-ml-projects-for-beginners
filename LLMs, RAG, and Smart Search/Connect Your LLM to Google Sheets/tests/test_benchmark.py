from __future__ import annotations

from pathlib import Path

from ai_spreadsheet_analytics.benchmark import BenchmarkRunner
from ai_spreadsheet_analytics.state_store import SQLiteStateStore


def test_benchmark_runner_loads_cases_and_scores(tmp_path: Path) -> None:
    db = SQLiteStateStore(tmp_path / "state.db")
    runner = BenchmarkRunner(db)

    cases = runner.load_cases(Path("data/benchmarks/questions.json"))
    assert len(cases) == 100

    def answer_fn(_question: str) -> tuple[str, float]:
        return "revenue trend with recommendation", 25.0

    run_id, results, aggregate = runner.run("qwen3.5:4b", cases[:5], answer_fn)

    assert run_id
    assert len(results) == 5
    assert aggregate["avg_latency_ms"] == 25.0
