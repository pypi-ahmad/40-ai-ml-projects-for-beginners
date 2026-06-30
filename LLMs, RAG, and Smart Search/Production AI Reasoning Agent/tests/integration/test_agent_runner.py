from __future__ import annotations

from reasoning_agent.agent.runner import AgentRunner
from reasoning_agent.schemas import ReasoningMode
from reasoning_agent.settings import Settings


def test_runner_works_without_ollama_by_fallback(tmp_path) -> None:
    settings = Settings(
        ollama_base_url="http://127.0.0.1:11434",
        log_dir=str(tmp_path / "logs"),
        chroma_dir=str(tmp_path / "chroma"),
        benchmark_dataset_path="benchmarks/prompts.jsonl",
    )
    runner = AgentRunner(settings=settings)
    try:
        response = runner.run("test-session", "Calculate 2 + 2", mode=ReasoningMode.REACT)
        assert response.answer
        assert response.iterations >= 0
    finally:
        runner.close()
