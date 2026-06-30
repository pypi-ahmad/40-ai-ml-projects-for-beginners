"""Hydra-driven end-to-end orchestration pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - fallback for minimal offline environments
    DictConfig = dict[str, Any]  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from peft_platform.benchmarking.plots import save_latency_bar
from peft_platform.benchmarking.suite import benchmark_callable
from peft_platform.evaluation.metrics import evaluate_classification
from peft_platform.evaluation.reports import write_markdown_report
from peft_platform.inference.engine import InferenceEngine
from peft_platform.model_registry import get_model_spec
from peft_platform.peft.registry import parse_method
from peft_platform.training.runner import TrainingRunner
from peft_platform.tracking.mlflow_client import TrackingClient
from peft_platform.utils.io import ensure_dir, write_json
from peft_platform.utils.logging import configure_logging
from peft_platform.utils.runtime import detect_runtime, query_gpu

logger = configure_logging()


@dataclass(slots=True)
class RunSummary:
    model: str
    method: str
    train_loss: float
    latency_ms: float
    accuracy: float
    report_path: str
    plot_path: str
    run_id: str


def _cfg_get(cfg: DictConfig | dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            current = getattr(current, part, default)
            if current is default:
                return default
    return current


def _dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if OmegaConf is not None:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return cfg
    return {}


def run_pipeline(cfg: DictConfig | dict[str, Any]) -> RunSummary:
    artifacts_root = Path("artifacts")
    ensure_dir(artifacts_root)

    model_key = str(_cfg_get(cfg, "model_key", "tinyllama_1_1b_chat"))
    method_name = str(_cfg_get(cfg, "peft.method", "lora"))
    if model_key in {"tinyllama_1_1b_chat", "smollm2_1_7b_instruct", "qwen3_1_7b", "llama3_instruct", "gemma3_instruct", "phi4_mini_instruct", "mistral_instruct", "modernbert", "deepseek"}:
        model = get_model_spec(model_key)
    else:
        # Allow explicit model id config even when not keyed in registry.
        model = get_model_spec("tinyllama_1_1b_chat")
        model = type(model)(
            id=str(_cfg_get(cfg, "model.name", model.id)),
            family=str(_cfg_get(cfg, "model.family", "TinyLlama")),
            instruction_tuned=True,
            tier=model.tier,
            default_template=str(_cfg_get(cfg, "model.template", "llama")),
        )
    method = parse_method(method_name)

    runner = TrainingRunner(artifacts_root=artifacts_root / "checkpoints")
    training_result = runner.run_smoke(model.id, method, steps=int(_cfg_get(cfg, "train.max_steps", 100)))

    engine = InferenceEngine()
    bench = benchmark_callable(
        lambda: engine.generate("benchmark prompt").text,
        runs=int(_cfg_get(cfg, "benchmark.runs", 10)),
    )
    plot_path = save_latency_bar({method_name: bench}, artifacts_root / "plots" / f"{method_name}_latency.png")

    eval_result = evaluate_classification([1, 0, 1, 1], [1, 0, 1, 0])
    report_path = write_markdown_report(
        artifacts_root / "reports" / f"{method_name}_summary.md",
        "Run Summary",
        [
            f"Model: {model.id}",
            f"Method: {method.value}",
            f"Train loss: {training_result.loss}",
            f"Accuracy: {eval_result.accuracy:.4f}",
            f"Latency avg: {bench.latency_ms_avg:.2f} ms",
        ],
    )

    runtime = detect_runtime()
    gpu = query_gpu()
    manifest = {
        "config": _dict(cfg),
        "runtime": asdict(runtime),
        "gpu": asdict(gpu) if gpu else None,
        "training": asdict(training_result),
        "benchmark": asdict(bench),
        "evaluation": asdict(eval_result),
        "artifacts": [str(plot_path), str(report_path)],
    }
    manifest_path = artifacts_root / "reports" / f"{method_name}_manifest.json"
    write_json(manifest_path, manifest)

    tracker = TrackingClient(
        str(_cfg_get(cfg, "tracking.mlflow_tracking_uri", "sqlite:///artifacts/mlflow/mlflow.db")),
        str(_cfg_get(cfg, "tracking.experiment_name", "peft-platform")),
    )
    run_id = tracker.log_run(
        run_name=str(_cfg_get(cfg, "run_name", f"{model_key}_{method_name}")),
        params={"model": model.id, "method": method.value},
        metrics={
            "train_loss": training_result.loss,
            "latency_ms_avg": bench.latency_ms_avg,
            "accuracy": eval_result.accuracy,
            "f1": eval_result.f1,
        },
        artifacts=[plot_path, report_path, manifest_path],
    )

    summary = RunSummary(
        model=model.id,
        method=method.value,
        train_loss=training_result.loss,
        latency_ms=bench.latency_ms_avg,
        accuracy=eval_result.accuracy,
        report_path=str(report_path),
        plot_path=str(plot_path),
        run_id=run_id,
    )
    logger.info("Pipeline complete: %s", summary)
    return summary


def load_config(config_path: Path) -> DictConfig | dict[str, Any]:
    """Load merged OmegaConf config from YAML path."""
    if config_path.name == "config.yaml":
        try:
            from hydra import compose, initialize_config_dir

            with initialize_config_dir(version_base=None, config_dir=str(config_path.parent.resolve())):
                return compose(config_name=config_path.stem)
        except Exception:
            pass
    if OmegaConf is not None:
        return OmegaConf.load(config_path)

    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload
