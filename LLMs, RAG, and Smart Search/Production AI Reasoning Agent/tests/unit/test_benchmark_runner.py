from __future__ import annotations

from pathlib import Path

from reasoning_agent.evaluation.benchmark import BenchmarkRunner
from reasoning_agent.evaluation.datasets import EvalPrompt
from reasoning_agent.settings import Settings


class _FakeDataset:
    def __init__(self):
        self.records = [
            EvalPrompt(prompt_id="1", category="math", prompt="2+2", expected_keywords=["4"]),
            EvalPrompt(prompt_id="2", category="reasoning", prompt="why", expected_keywords=["because"]),
        ]


class _FakeResp:
    def __init__(self, answer: str):
        self.answer = answer
        self.success = True
        self.termination_reason = "completed"
        self.trace = []
        self.metrics = {"retries": 0}


class _FakeAgentRunner:
    def __init__(self, settings):
        self.settings = settings

    def run(self, session_id: str, user_input: str, mode):
        return _FakeResp("because 4")

    def close(self):
        return None


class _FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def available_models(self):
        return ["qwen3:8b", "granite4.1:3b"]

    def generate(self, *args, **kwargs):
        class R:
            text = '{"reasoning_quality":0.7,"correctness":0.8,"grounding":0.6,"completeness":0.7,"tool_usage":0.5,"rationale":"ok"}'

        return R()

    def close(self):
        return None


def test_benchmark_runner_with_fakes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("reasoning_agent.evaluation.benchmark.BenchmarkDataset.load_jsonl", lambda path: _FakeDataset())
    monkeypatch.setattr("reasoning_agent.evaluation.benchmark.AgentRunner", _FakeAgentRunner)
    monkeypatch.setattr("reasoning_agent.evaluation.benchmark.OllamaClient", _FakeLLM)

    settings = Settings(
        benchmark_dataset_path="benchmarks/prompts.jsonl",
        benchmark_output_dir=str(tmp_path),
        compare_models="granite4.1:3b,llama3.1:8b",
    )

    out = BenchmarkRunner(settings).run(output_dir=tmp_path)
    assert out["summary"]
    assert (tmp_path / "benchmark_summary.json").exists()
