from pathlib import Path

from llmft.config.loader import load_project_config
from llmft.pipeline import ProjectRunner


def test_project_runner_end_to_end_dry_run(tmp_path) -> None:
    import os

    os.environ["LLMFT_TRANSFORMERS_MODEL"] = "sshleifer/tiny-gpt2"
    os.environ["LLMFT_TRAIN_MODEL"] = "sshleifer/tiny-gpt2"
    config = load_project_config(Path("configs/project.yaml"))
    config.runtime.artifacts_dir = str(tmp_path / "artifacts")
    config.runtime.cache_dir = str(tmp_path / "cache")
    config.data.datasets = ["alpaca_cleaned"]
    config.data.max_samples_per_dataset = 8
    config.inference.vllm_host = "http://127.0.0.1:9"
    config.inference.ollama_host = "http://127.0.0.1:9"
    config.inference.benchmark_backends = ["transformers"]

    runner = ProjectRunner(config)

    env_report = runner.validate_env()
    bundle = runner.build_data()
    train = runner.train_sft(dry_run=False)
    eval_path = runner.run_evaluation()
    bench_path = runner.run_benchmark()
    infer_path = runner.run_inference("Explain LoRA.")
    export_path = runner.run_export()
    telemetry_path = runner.telemetry_snapshot()

    assert env_report.exists()
    assert bundle.manifest_path.exists()
    assert train.eval_loss > 0
    assert eval_path.exists()
    assert bench_path.exists()
    assert (bench_path.parent / "latency_benchmark.csv").exists()
    assert (bench_path.parent / "latency_benchmark.md").exists()
    assert infer_path.exists()
    assert export_path.exists()
    assert telemetry_path.exists()
