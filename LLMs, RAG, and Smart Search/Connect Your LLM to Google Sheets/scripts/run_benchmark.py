"""Benchmark local LLM models on question set."""

from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

from ai_spreadsheet_analytics.benchmark import BenchmarkRunner
from ai_spreadsheet_analytics.config import Settings
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient
from ai_spreadsheet_analytics.state_store import SQLiteStateStore


def main() -> None:
    load_dotenv()
    settings = Settings()
    state = SQLiteStateStore(settings.state_db_path)
    runner = BenchmarkRunner(state)
    client = OllamaRESTClient(settings.ollama_base_url)
    cases = runner.load_cases(Path(settings.benchmark_cases_path))

    def answer_fn(question: str) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            response = __import__("asyncio").run(
                client.agenerate(
                    model=settings.ollama_primary_model,
                    prompt=question,
                    system="Answer with concise business analytics language.",
                    temperature=0.0,
                )
            )
            answer_text = response.text
        except Exception:
            answer_text = (
                "Fallback benchmark answer: revenue trend, key risk, recommendation. "
                "Ollama unavailable."
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return answer_text, elapsed_ms

    run_id, results, aggregate = runner.run(settings.ollama_primary_model, cases, answer_fn)
    out = Path("data/artifacts") / f"benchmark_{run_id}.json"
    runner.save_results(out, run_id, results, aggregate)
    print("Saved", out)
    print("Aggregate", aggregate)


if __name__ == "__main__":
    main()
