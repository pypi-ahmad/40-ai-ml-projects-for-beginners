from unittest.mock import MagicMock, patch

import pytest

from src.benchmarking import MODELS, PROMPT_LENGTHS, BenchmarkRunner
from src.visualization import BenchmarkVisualizer


@pytest.fixture
def runner() -> BenchmarkRunner:
    r = BenchmarkRunner()
    yield r
    r.close()


def test_prompt_lengths_keys() -> None:
    assert set(PROMPT_LENGTHS.keys()) == {"short", "medium", "long"}


def test_prompt_lengths_non_empty() -> None:
    for k, v in PROMPT_LENGTHS.items():
        assert len(v) > 0, f"{k} prompt is empty"


def test_default_models() -> None:
    assert len(MODELS) == 4
    assert all(":" in m for m in MODELS)
    assert "qwen3.5:4b" in MODELS


@patch("src.benchmarking.OllamaClient")
def test_run_all_structure(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.measure_inference_time.return_value = {
        "latency_s": 0.5,
        "tokens": 42,
        "full_result": {"response": "ok"},
    }
    r = BenchmarkRunner(models=["qwen3.5:2b", "qwen3.5:4b"])
    results = r.run_all({"short": "test prompt"})
    assert "qwen3.5:2b" in results
    assert "qwen3.5:4b" in results
    assert "short" in results["qwen3.5:2b"]
    assert results["qwen3.5:2b"]["short"]["latency_s"] == 0.5
    assert results["qwen3.5:2b"]["short"]["tokens"] == 42
    r.close()


@patch("src.benchmarking.OllamaClient")
def test_run_all_calls_measure_for_each(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.measure_inference_time.return_value = {
        "latency_s": 0.1,
        "tokens": 10,
        "full_result": {"response": "ok"},
    }
    r = BenchmarkRunner(models=["qwen3.5:2b"])
    r.run_all({"a": "p1", "b": "p2"})
    assert mock_ollama.return_value.measure_inference_time.call_count == 2
    r.close()


def test_custom_models() -> None:
    r = BenchmarkRunner(models=["custom:1b"])
    assert r.models == ["custom:1b"]
    r.close()


SAMPLE_DATA = {
    "model_a": {
        "Short": {"latency_s": 0.5, "tokens": 50},
        "Medium": {"latency_s": 1.0, "tokens": 100},
        "Long": {"latency_s": 2.0, "tokens": 200},
    },
    "model_b": {
        "Short": {"latency_s": 0.3, "tokens": 30},
        "Medium": {"latency_s": 0.7, "tokens": 75},
        "Long": {"latency_s": 1.5, "tokens": 150},
    },
}


@pytest.fixture
def visualizer() -> BenchmarkVisualizer:
    v = BenchmarkVisualizer(figs_dir="/tmp/test_bench_figs")
    import os

    os.makedirs(v.figs_dir, exist_ok=True)
    yield v


def test_latency_chart(visualizer) -> None:
    path = visualizer.latency_chart(SAMPLE_DATA)
    assert path.endswith(".png")


def test_throughput_chart(visualizer) -> None:
    path = visualizer.throughput_chart(SAMPLE_DATA)
    assert path.endswith(".png")


def test_prompt_scale_chart(visualizer) -> None:
    path = visualizer.prompt_scale_chart(SAMPLE_DATA)
    assert path.endswith(".png")


def test_radar_chart(visualizer) -> None:
    path = visualizer.radar_chart(SAMPLE_DATA)
    assert path.endswith(".png")


def test_generate_all(visualizer) -> None:
    paths = visualizer.generate_all(SAMPLE_DATA)
    assert set(paths.keys()) == {"latency", "throughput", "prompt_scale", "radar"}
    for v in paths.values():
        assert v.endswith(".png")
