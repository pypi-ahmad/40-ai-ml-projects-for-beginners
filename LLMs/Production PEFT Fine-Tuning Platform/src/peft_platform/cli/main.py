"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from peft_platform.benchmarking.plots import save_latency_bar
from peft_platform.benchmarking.suite import benchmark_callable
from peft_platform.data.processing import clean_samples, compute_stats, deduplicate_samples, split_samples
from peft_platform.data.schemas import Sample
from peft_platform.inference.engine import InferenceEngine
from peft_platform.model_registry import get_model_spec, list_models
from peft_platform.pipeline import load_config, run_pipeline
from peft_platform.peft.adapters import AdapterManager, AdapterRecord
from peft_platform.peft.registry import parse_method
from peft_platform.training.runner import TrainingRunner
from peft_platform.utils.logging import configure_logging
from peft_platform.utils.runtime import detect_runtime, query_gpu

app = typer.Typer(help="Production PEFT Platform CLI")
logger = configure_logging()


@app.command("runtime")
def runtime_command() -> None:
    runtime = detect_runtime()
    gpu = query_gpu()
    typer.echo(f"python={runtime.python_version} device={runtime.device} cuda={runtime.cuda_available}")
    if gpu is not None:
        typer.echo(f"gpu={gpu.name} memory_mb={gpu.memory_total_mb} driver={gpu.driver_version}")


@app.command("models")
def models_command() -> None:
    for model in list_models():
        typer.echo(f"{model.id} | family={model.family} | tier={model.tier.value}")


@app.command("dataset-smoke")
def dataset_smoke_command() -> None:
    samples = [
        Sample(task_type="instruction", instruction="Summarize", input="Text", output="Summary"),
        Sample(task_type="instruction", instruction="Summarize", input="Text", output="Summary"),
        Sample(task_type="qa", instruction="Answer", input="What is PEFT?", output="Efficient tuning"),
    ]
    cleaned = clean_samples(samples)
    deduped = deduplicate_samples(cleaned)
    splits = split_samples(deduped, val_size=0.2, test_size=0.2, seed=42)
    stats = compute_stats(deduped)
    typer.echo(f"dataset size={stats.size} unique_ratio={stats.unique_ratio:.3f}")
    typer.echo(f"split train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")


@app.command("train")
def train_command(model: str = "tinyllama_1_1b_chat", method: str = "lora", steps: int = 8) -> None:
    spec = get_model_spec(model)
    peft_method = parse_method(method)
    runner = TrainingRunner(artifacts_root=Path("artifacts/checkpoints"))
    result = runner.run_smoke(model_id=spec.id, method=peft_method, steps=steps)
    typer.echo(f"trained method={result.method} model={result.model_id} loss={result.loss} steps={result.steps}")


@app.command("live-e2e")
def live_e2e_command(
    model_id: str = "sshleifer/tiny-gpt2",
    max_steps: int = 10,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
) -> None:
    runner = TrainingRunner(artifacts_root=Path("artifacts/live_runs"))
    result = runner.run_live_lora(
        model_id=model_id,
        output_name="lora_live_e2e",
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
    )
    typer.echo(
        f"live training done model={result.model_id} steps={result.train_steps} "
        f"loss={result.train_loss:.4f} adapter_dir={result.adapter_dir}"
    )
    typer.echo(f"generation_sample={result.generated_text[:200]}")


@app.command("infer")
def infer_command(prompt: str, model_id: str = "mock-model") -> None:
    engine = InferenceEngine(model_id=model_id)
    engine.load()
    output = engine.generate(prompt=prompt)
    typer.echo(output.text)


@app.command("benchmark")
def benchmark_command() -> None:
    engine = InferenceEngine()
    result = benchmark_callable(lambda: engine.generate("hello benchmark").text, runs=20)
    chart = save_latency_bar({"mock": result}, Path("artifacts/plots/latency_bar.png"))
    typer.echo(
        f"lat_avg={result.latency_ms_avg:.2f}ms p95={result.latency_ms_p95:.2f}ms "
        f"thr={result.throughput_req_per_sec:.2f}/s peak_mem={result.peak_memory_mb:.2f}MB chart={chart}"
    )


@app.command("run-pipeline")
def run_pipeline_command(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(Path(config_path))
    summary = run_pipeline(cfg)
    typer.echo(
        f"pipeline done model={summary.model} method={summary.method} "
        f"loss={summary.train_loss} acc={summary.accuracy:.3f} run_id={summary.run_id}"
    )


@app.command("adapter-add")
def adapter_add_command(name: str, method: str, base_model: str, path: str) -> None:
    manager = AdapterManager(Path("artifacts/adapter_registry.json"))
    manager.add_adapter(AdapterRecord(name=name, method=method, base_model=base_model, path=path))
    typer.echo(f"adapter saved: {name}")


@app.command("adapter-list")
def adapter_list_command() -> None:
    manager = AdapterManager(Path("artifacts/adapter_registry.json"))
    for item in manager.list_adapters():
        typer.echo(f"{item.name} | {item.method} | {item.base_model} | merged={item.merged}")


if __name__ == "__main__":
    app()
