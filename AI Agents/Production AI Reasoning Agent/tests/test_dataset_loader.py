from __future__ import annotations

from pathlib import Path

from reasoning_agent.evals.dataset import load_benchmark_prompts


def test_load_benchmark_prompts(tmp_path: Path) -> None:
    file = tmp_path / "bench.jsonl"
    file.write_text(
        '{"id":"p1","category":"math","prompt":"2+2","expected_keywords":["4"],"required_tools":["calculator"],"current_events":false}\n',
        encoding="utf-8",
    )

    prompts = load_benchmark_prompts(file)

    assert len(prompts) == 1
    assert prompts[0].prompt_id == "p1"
    assert prompts[0].required_tools == ["calculator"]
