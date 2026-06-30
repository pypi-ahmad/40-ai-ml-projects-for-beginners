"""End-to-end project orchestration."""

from __future__ import annotations

import asyncio
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from llmft.config.loader import load_project_config
from llmft.config.schemas import ProjectConfig
from llmft.data.pipeline import DatasetBundle, DatasetPipeline
from llmft.evaluation import write_benchmark_tables
from llmft.evaluation.evaluator import EvaluationEngine
from llmft.export.manager import ExportManager
from llmft.inference.backends import InferenceRouter
from llmft.models.registry import ModelRegistry, ModelResolution
from llmft.monitoring import MLflowTracker, RuntimeMonitor
from llmft.security.checks import detect_unsafe_response, validate_prompt
from llmft.training.engine import TrainingEngine
from llmft.training.types import HPOReport, TrainingReport
from llmft.utils.io import ensure_dir, write_json
from llmft.utils.logging import get_logger
from llmft.utils.seed import set_seed


class ProjectRunner:
    """Coordinates data, training, evaluation, inference, export, and reporting."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.artifacts_dir = ensure_dir(config.runtime.artifacts_dir)
        self.logger = get_logger("llmft.runner", self.artifacts_dir / "logs" / "runner.log")
        self.templates = None
        self.dataset_pipeline = DatasetPipeline(cache_dir=config.runtime.cache_dir, seed=config.runtime.seed)
        self.model_registry = ModelRegistry()
        self.monitor = RuntimeMonitor()
        self.tracker = MLflowTracker(self.artifacts_dir, experiment_name=config.name)
        self._bundle: DatasetBundle | None = None
        self._train_report: TrainingReport | None = None
        self._hpo_report: HPOReport | None = None

    @classmethod
    def from_config_path(cls, path: str | Path) -> "ProjectRunner":
        """Build runner from YAML config path."""
        return cls(load_project_config(path))

    def validate_env(self) -> Path:
        """Validate local environment and write report."""
        report = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seed": self.config.runtime.seed,
            "cuda_visible": self._check_cuda_visible(),
            "timestamp": time.time(),
        }
        out = self.artifacts_dir / "reports" / "env_report.json"
        write_json(out, report)
        self.logger.info("environment validated")
        return out

    def build_data(self) -> DatasetBundle:
        """Run dataset pipeline and store bundle."""
        set_seed(self.config.runtime.seed)
        bundle = self.dataset_pipeline.build(self.config.data, self.artifacts_dir)
        self._bundle = bundle
        self.logger.info("dataset ready version=%s", bundle.version_id)
        return bundle

    def train_sft(self, dry_run: bool = False) -> TrainingReport:
        """Run SFT/PEFT training pipeline."""
        bundle = self._bundle or self.build_data()
        resolution = self._resolve_primary_model()

        trainer = TrainingEngine(self.artifacts_dir)
        report = trainer.train_sft(self.config.train, bundle, resolution, dry_run=dry_run)
        self._train_report = report

        if self.config.hpo.enabled:
            self._hpo_report = trainer.run_hpo(self.config.hpo, self.config.train, report)

        write_json(
            self.artifacts_dir / "reports" / "training_latest.json",
            asdict(report) | {"checkpoints_dir": str(report.checkpoints_dir)},
        )
        self.tracker.log_run(
            run_name=report.run_id,
            params={
                "model_alias": report.model_alias,
                "model_id": report.model_id,
                "peft_method": report.peft_method,
                "train_samples": report.train_samples,
            },
            metrics={
                "train_loss": report.train_loss,
                "eval_loss": report.eval_loss,
                "steps": float(report.steps),
            },
            artifacts={"training": str(self.artifacts_dir / "reports" / "training_latest.json")},
        )
        return report

    def run_evaluation(self) -> Path:
        """Run evaluation pipeline."""
        bundle = self._bundle or self.build_data()
        router = InferenceRouter(self.config.inference)
        engine = EvaluationEngine(self.artifacts_dir, router)
        report = asyncio.run(engine.evaluate(self.config.evaluation, bundle))
        payload = json.loads(report.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        self.tracker.log_run(
            run_name=f"eval-{bundle.version_id}",
            params={"judge_model": self.config.evaluation.judge_model},
            metrics={key: float(value) for key, value in metrics.items() if isinstance(value, int | float)},
            artifacts={"evaluation": str(report)},
        )
        return report

    def run_benchmark(self) -> Path:
        """Run latency benchmarks on configured real backends."""
        prompts = [
            "Summarize benefits of QLoRA in 2 sentences.",
            "Write Python function to add two numbers.",
            "Explain train/val/test split importance.",
            "Translate 'hello world' to Arabic.",
        ]
        backends = list(self.config.inference.benchmark_backends)
        if not backends:
            raise RuntimeError("No benchmark backends configured")

        results: list[dict[str, float | str | int]] = []
        for backend in backends:
            cfg = self.config.inference
            cfg.backend = backend
            router = InferenceRouter(cfg)
            outcome = asyncio.run(router.benchmark(prompts))
            results.append(asdict(outcome))

        out = self.artifacts_dir / "benchmarks" / "latency_benchmark.json"
        write_json(out, {"results": results})
        write_benchmark_tables(results, out.parent)
        metrics = {
            f"{item['backend']}_latency_ms": float(item["mean_latency_ms"])
            for item in results
            if "backend" in item and "mean_latency_ms" in item
        }
        self.tracker.log_run(
            run_name="benchmark-latency",
            params={"prompt_count": len(prompts)},
            metrics=metrics,
            artifacts={"benchmark": str(out)},
        )
        return out

    def run_inference(self, prompt: str) -> Path:
        """Run one inference request with safety checks."""
        ok, reason = validate_prompt(prompt, self.config.safety)
        if not ok:
            raise ValueError(f"Prompt rejected: {reason}")

        router = InferenceRouter(self.config.inference)
        response = asyncio.run(router.generate(prompt))
        unsafe, unsafe_reason = detect_unsafe_response(response, self.config.safety)

        out = self.artifacts_dir / "reports" / "inference_latest.json"
        write_json(
            out,
            {
                "prompt": prompt,
                "response": response,
                "unsafe": unsafe,
                "unsafe_reason": unsafe_reason,
                "backend": self.config.inference.backend,
            },
        )
        return out

    def run_export(self) -> Path:
        """Run model export pipeline."""
        report = self._train_report or self.train_sft(dry_run=False)
        manager = ExportManager(self.config.export)
        out = manager.run(report)
        return out

    def telemetry_snapshot(self) -> Path:
        """Capture runtime telemetry snapshot."""
        snapshot = self.monitor.capture()
        out = self.artifacts_dir / "reports" / "runtime_snapshot.json"
        self.monitor.write(out, snapshot)
        return out

    def _resolve_primary_model(self) -> ModelResolution:
        target_alias = self.config.model.targets[0]
        resolution = self.model_registry.resolve(
            target_alias,
            available_model_ids=None,
            allow_fallback=self.config.model.allow_fallback,
        )
        write_json(
            self.artifacts_dir / "reports" / "model_resolution.json",
            {
                "alias": resolution.alias,
                "selected_id": resolution.selected_id,
                "used_fallback": resolution.used_fallback,
                "reason": resolution.reason,
            },
        )
        return resolution

    @staticmethod
    def _check_cuda_visible() -> bool:
        import shutil

        return shutil.which("nvidia-smi") is not None
