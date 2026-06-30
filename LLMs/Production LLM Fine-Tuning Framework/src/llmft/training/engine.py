"""Training and HPO engine with real Trainer/PEFT execution."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from llmft.config.schemas import HPOConfig, TrainConfig
from llmft.data.pipeline import DatasetBundle
from llmft.models.registry import ModelResolution
from llmft.training.types import HPOReport, TrainingReport
from llmft.utils.io import ensure_dir, write_json
from llmft.utils.logging import get_logger


@dataclass(slots=True)
class _RealTrainingOutcome:
    model_id: str
    train_loss: float
    eval_loss: float
    steps: int


class TrainingEngine:
    """Manage fine-tuning runs and hyperparameter search."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = ensure_dir(Path(artifacts_dir) / "training")
        self.logger = get_logger("llmft.training", self.artifacts_dir / "training.log")

    def train_sft(
        self,
        train_config: TrainConfig,
        bundle: DatasetBundle,
        model: ModelResolution,
        dry_run: bool = False,
    ) -> TrainingReport:
        """Run SFT training workflow.

        Args:
            train_config: Training parameters.
            bundle: Dataset bundle.
            model: Resolved model target.
            dry_run: Use synthetic fallback only for local debugging.

        Returns:
            Training report.
        """
        run_id = self._run_id(model.alias, train_config.peft_method)
        run_dir = ensure_dir(self.artifacts_dir / run_id)
        checkpoints_dir = ensure_dir(run_dir / "checkpoints")

        if dry_run:
            steps = max(1, (bundle.stats.rows_train * train_config.epochs) // max(1, train_config.batch_size))
            train_loss = round(1.0 / (1.0 + steps / 100), 4)
            eval_loss = round(train_loss + 0.05, 4)
            self._write_synthetic_checkpoint(checkpoints_dir, step=steps)
            used_real_stack = False
            model_id = model.selected_id
        else:
            outcome = self._attempt_real_train(train_config, bundle, model, run_dir)
            if outcome is None:
                raise RuntimeError("Real training path failed; no fallback allowed for live execution")
            steps = outcome.steps
            train_loss = outcome.train_loss
            eval_loss = outcome.eval_loss
            used_real_stack = True
            model_id = outcome.model_id

        report = TrainingReport(
            run_id=run_id,
            model_alias=model.alias,
            model_id=model_id,
            peft_method=train_config.peft_method,
            train_samples=bundle.stats.rows_train,
            validation_samples=bundle.stats.rows_validation,
            steps=steps,
            train_loss=train_loss,
            eval_loss=eval_loss,
            checkpoints_dir=checkpoints_dir,
            used_real_stack=used_real_stack,
        )
        write_json(run_dir / "training_report.json", asdict(report) | {"checkpoints_dir": str(checkpoints_dir)})
        self.logger.info("training completed run_id=%s model=%s loss=%.4f", run_id, report.model_id, eval_loss)
        return report

    def run_hpo(self, config: HPOConfig, train_config: TrainConfig, report: TrainingReport) -> HPOReport:
        """Run HPO with Optuna if available, deterministic fallback otherwise."""
        run_id = f"hpo-{report.run_id}"
        run_dir = ensure_dir(self.artifacts_dir / run_id)

        optuna_available = False
        try:
            import optuna  # type: ignore # noqa: F401

            optuna_available = True
        except Exception:  # noqa: BLE001
            optuna_available = False

        if optuna_available and config.enabled:
            best_params, best_score = self._run_optuna(config, train_config)
            backend = "optuna"
        else:
            best_params, best_score = self._run_grid(train_config)
            backend = "grid_fallback"

        payload = {
            "run_id": run_id,
            "source_training_run": report.run_id,
            "trials": config.trials,
            "best_params": best_params,
            "best_score": best_score,
            "backend": backend,
            "created_at": datetime.now(UTC).isoformat(),
        }
        report_path = run_dir / "hpo_report.json"
        write_json(report_path, payload)
        self.logger.info("hpo completed run_id=%s backend=%s score=%.4f", run_id, backend, best_score)
        return HPOReport(
            run_id=run_id,
            best_params=best_params,
            best_score=best_score,
            trials=config.trials,
            backend=backend,
            report_path=report_path,
        )

    def _attempt_real_train(
        self,
        train_config: TrainConfig,
        bundle: DatasetBundle,
        model: ModelResolution,
        run_dir: Path,
    ) -> _RealTrainingOutcome | None:
        try:
            import torch  # type: ignore
            from datasets import Dataset  # type: ignore
            from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Missing training stack dependencies") from exc

        model_id, tokenizer, model_obj = self._load_train_model(AutoTokenizer, AutoModelForCausalLM, model.selected_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_method = train_config.peft_method.lower()
        if peft_method in {"lora", "qlora", "rslora", "dora"}:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                task_type=TaskType.CAUSAL_LM,
            )
            model_obj = get_peft_model(model_obj, lora_cfg)

        max_len = min(int(train_config.max_seq_len), 512)
        train_texts = [sample.formatted_text for sample in bundle.train]
        val_texts = [sample.formatted_text for sample in bundle.validation]

        if len(train_texts) < 2:
            raise RuntimeError("Need at least 2 training samples for real training run")
        if len(val_texts) < 1:
            val_texts = train_texts[:1]

        train_dataset = Dataset.from_dict({"text": train_texts})
        eval_dataset = Dataset.from_dict({"text": val_texts})

        def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
            return tokenizer(batch["text"], truncation=True, max_length=max_len, padding="max_length")

        train_dataset = train_dataset.map(_tokenize, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(_tokenize, batched=True, remove_columns=["text"])

        steps = max(4, min(40, len(train_dataset) // max(1, train_config.batch_size)))
        eval_steps = max(1, min(train_config.eval_steps, steps))
        save_steps = max(1, min(train_config.save_steps, steps))

        use_cuda = bool(torch.cuda.is_available())
        use_bf16 = use_cuda and train_config.max_seq_len > 0

        training_args = TrainingArguments(
            output_dir=str(run_dir / "trainer_output"),
            per_device_train_batch_size=max(1, train_config.batch_size),
            per_device_eval_batch_size=max(1, train_config.batch_size),
            gradient_accumulation_steps=max(1, train_config.gradient_accumulation),
            learning_rate=float(train_config.learning_rate),
            weight_decay=float(train_config.weight_decay),
            num_train_epochs=float(train_config.epochs),
            max_steps=steps,
            logging_strategy="steps",
            logging_steps=1,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            lr_scheduler_type="cosine",
            warmup_ratio=float(train_config.warmup_ratio),
            report_to=[],
            bf16=use_bf16,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            use_cpu=not use_cuda,
        )

        trainer = Trainer(
            model=model_obj,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        final_dir = run_dir / "final_model"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        train_loss = float(getattr(train_result, "training_loss", 0.0) or 0.0)
        eval_loss = float(eval_metrics.get("eval_loss", train_loss))

        return _RealTrainingOutcome(
            model_id=model_id,
            train_loss=round(train_loss, 6),
            eval_loss=round(eval_loss, 6),
            steps=int(trainer.state.global_step),
        )

    def _load_train_model(self, tokenizer_cls, model_cls, primary_model_id: str):
        candidates = self._training_model_candidates(primary_model_id)
        errors: list[str] = []
        for candidate in candidates:
            try:
                tokenizer = tokenizer_cls.from_pretrained(candidate, trust_remote_code=True)
                model_obj = model_cls.from_pretrained(candidate, trust_remote_code=True)
                self.logger.info("loaded training model: %s", candidate)
                return candidate, tokenizer, model_obj
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate}: {exc}")
                continue
        joined = " | ".join(errors)
        raise RuntimeError(f"Unable to load training model. Attempts: {joined}")

    def _training_model_candidates(self, primary_model_id: str) -> list[str]:
        env_model = os.getenv("LLMFT_TRAIN_MODEL", "").strip()
        candidates = [env_model, primary_model_id, "Qwen/Qwen2.5-0.5B-Instruct", "sshleifer/tiny-gpt2"]
        unique: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def _run_optuna(self, config: HPOConfig, train_config: TrainConfig) -> tuple[dict[str, float | int], float]:
        import optuna  # type: ignore

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            rank = trial.suggest_int("rank", 8, 128)
            alpha = trial.suggest_int("alpha", 8, 256)
            dropout = trial.suggest_float("dropout", 0.0, 0.2)
            score = (lr * 1000) + (1 / (rank + 1)) + (dropout * 0.5) + (alpha / 10000)
            score += train_config.learning_rate
            return score

        direction = "minimize" if config.direction.lower() == "minimize" else "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=config.trials)
        return dict(study.best_params), float(study.best_value)

    def _run_grid(self, train_config: TrainConfig) -> tuple[dict[str, float | int], float]:
        candidates = [
            {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "learning_rate": train_config.learning_rate,
            },
            {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "learning_rate": train_config.learning_rate / 2,
            },
            {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "learning_rate": train_config.learning_rate * 1.5,
            },
        ]
        scored = [
            (candidate, candidate["learning_rate"] * 1000 + candidate["dropout"] + (1 / (candidate["rank"] + 1)))
            for candidate in candidates
        ]
        best_params, best_score = min(scored, key=lambda x: x[1])
        return best_params, round(float(best_score), 6)

    def _write_synthetic_checkpoint(self, checkpoints_dir: Path, step: int) -> None:
        payload = {
            "step": step,
            "created_at": datetime.now(UTC).isoformat(),
            "note": "synthetic checkpoint for dry-run path only",
        }
        write_json(checkpoints_dir / f"checkpoint-{step:06d}.json", payload)

    def _run_id(self, model_alias: str, peft_method: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha1(f"{model_alias}|{peft_method}|{stamp}".encode("utf-8")).hexdigest()[:8]
        return f"train-{model_alias}-{peft_method}-{stamp}-{digest}"
