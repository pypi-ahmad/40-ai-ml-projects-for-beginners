"""CLI entrypoint for domain fine-tuning framework."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from domain_llm_ft.benchmark.report import benchmark_table, render_benchmark_charts
from domain_llm_ft.benchmark.runner import benchmark_matrix
from domain_llm_ft.compression.exporters import export_onnx, export_safetensors, export_torchscript
from domain_llm_ft.config.loader import load_config
from domain_llm_ft.data.loader import DatasetLoader
from domain_llm_ft.hpo.optuna_runner import OptunaRunner
from domain_llm_ft.inference.engine import InferenceEngine
from domain_llm_ft.models.registry import load_sequence_classifier
from domain_llm_ft.tokenization.factory import load_tokenizer
from domain_llm_ft.training.matrix import run_public_matrix
from domain_llm_ft.training.pipeline import run_training
from domain_llm_ft.utils.io import write_json
from domain_llm_ft.utils.logging import configure_logging

app = typer.Typer(help="Production Domain LLM Fine-Tuning Framework")
console = Console()


@app.command("prepare-data")
def prepare_data(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Run data loading and EDA pipeline."""
    cfg = load_config(config_path)
    configure_logging(cfg.paths.logs_dir)
    try:
        dataset = DatasetLoader(cfg.dataset).load()
    except Exception as exc:
        console.print(f"[red]prepare-data failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    payload = {split: len(dataset[split]) for split in dataset.keys()}
    write_json(Path(cfg.paths.artifacts_dir) / "reports" / "dataset_splits.json", payload)
    console.print("[green]Data preparation complete[/green]")


@app.command("train")
def train(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Run end-to-end training pipeline."""
    cfg = load_config(config_path)
    configure_logging(cfg.paths.logs_dir)
    try:
        artifacts = run_training(cfg)
    except Exception as exc:
        console.print(f"[red]train failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(artifacts))


@app.command("tune")
def tune(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Run Optuna tuning."""
    cfg = load_config(config_path)
    study = OptunaRunner(cfg).run(Path(cfg.paths.artifacts_dir) / "optuna.db")
    payload = {
        "best_params": study.best_trial.params,
        "best_value": study.best_value,
    }
    write_json(Path(cfg.paths.artifacts_dir) / "reports" / "optuna_summary.json", payload)
    console.print_json(json.dumps(payload))


@app.command("benchmark")
def benchmark(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Run benchmark matrix and generate reports."""
    cfg = load_config(config_path)
    samples = [
        "Customer asks for refund because shipment delayed.",
        "Bank analyst upgrades stock target after earnings.",
        "Medical report indicates elevated blood pressure.",
        "Legal contract includes indemnification clause.",
    ] * 16

    models = [
        "distilbert",
        "bert_base",
        "roberta_base",
        "deberta_v3",
        "modernbert",
        "minilm",
        "e5",
        "qwen",
        "gemma",
        "phi",
        "tinyllama",
    ]
    results = benchmark_matrix(models, samples)
    if not results:
        console.print("[red]benchmark failed:[/red] no benchmarkable models found in local cache")
        raise typer.Exit(code=1)
    table = benchmark_table(results)

    reports_dir = Path(cfg.paths.artifacts_dir) / "reports"
    figures_dir = Path(cfg.paths.artifacts_dir) / "figures" / "benchmark"
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "benchmark.csv", index=False)
    render_benchmark_charts(table, figures_dir)
    console.print(f"Benchmark complete: {reports_dir / 'benchmark.csv'}")


@app.command("export")
def export(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Export model in deployment formats."""
    cfg = load_config(config_path)
    model = load_sequence_classifier(cfg.model.name, cfg.model.num_labels)
    tokenizer = load_tokenizer(cfg.tokenizer.name)
    export_dir = Path(cfg.paths.artifacts_dir) / "exports"
    outputs = {
        "onnx": str(export_onnx(model, tokenizer, export_dir)),
        "torchscript": str(export_torchscript(model, tokenizer, export_dir)),
        "safetensors": str(export_safetensors(model, export_dir)),
    }
    write_json(export_dir / "manifest.json", outputs)
    console.print_json(json.dumps(outputs))


@app.command("infer")
def infer(text: str, model_name: str = "distilbert-base-uncased") -> None:
    """Run single inference from CLI."""
    prediction = InferenceEngine(model_name=model_name).predict(text)
    console.print_json(
        json.dumps(
            {
                "label": prediction.label,
                "score": prediction.score,
                "probabilities": prediction.probabilities,
            }
        )
    )


@app.command("serve-api")
def serve_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve FastAPI app."""
    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-m",
            "uvicorn",
            "domain_llm_ft.serving.api:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=True,
    )


@app.command("serve-ui")
def serve_ui(port: int = 8501) -> None:
    """Serve Streamlit app."""
    subprocess.run(
        [
            str(Path(".venv/bin/python")),
            "-m",
            "streamlit",
            "run",
            "src/domain_llm_ft/ui/streamlit_app.py",
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
        ],
        check=True,
    )


@app.command("run-matrix")
def run_matrix(config_path: Path = Path("configs/baseline.yaml"), execute: bool = False) -> None:
    """Run or plan public model-dataset matrix."""
    cfg = load_config(config_path)
    table = run_public_matrix(cfg, execute=execute)
    out = Path(cfg.paths.artifacts_dir) / "reports" / ("matrix_results.csv" if execute else "matrix_plan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    console.print(f"Matrix {'execution' if execute else 'plan'} written: {out}")


@app.command("run-notebook")
def run_notebook(notebook_path: Path = Path("notebooks/01_zero_to_hero.ipynb")) -> None:
    """Execute notebook sequentially."""
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook_path),
            "--output",
            str(notebook_path.with_name(notebook_path.stem + ".executed.ipynb")),
        ],
        check=True,
    )


@app.command("collect-evidence")
def collect_evidence(config_path: Path = Path("configs/baseline.yaml")) -> None:
    """Bundle known artifact references into one manifest."""
    cfg = load_config(config_path)
    root = Path(cfg.paths.artifacts_dir)
    manifest = {
        "reports": [str(x) for x in sorted((root / "reports").glob("**/*")) if x.is_file()],
        "figures": [str(x) for x in sorted((root / "figures").glob("**/*")) if x.is_file()],
        "screenshots": [str(x) for x in sorted((root / "screenshots").glob("**/*")) if x.is_file()],
    }
    out = root / "reports" / "evidence_manifest.json"
    write_json(out, manifest)
    console.print(f"Evidence manifest: {out}")


if __name__ == "__main__":
    app()
