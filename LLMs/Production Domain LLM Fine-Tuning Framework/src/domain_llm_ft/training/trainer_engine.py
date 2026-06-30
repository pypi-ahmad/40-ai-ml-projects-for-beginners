"""Trainer-based training engine."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
from datasets import DatasetDict
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.evaluation.metrics import classification_metrics
from domain_llm_ft.models.registry import load_sequence_classifier, resolve_model_name
from domain_llm_ft.peft.adapters import apply_peft


class TrainerEngine:
    """Orchestrate fine-tuning with Hugging Face Trainer."""

    def __init__(self, config: ExperimentConfig, tokenizer, data_collator):
        self.config = config
        self.tokenizer = tokenizer
        self.data_collator = data_collator

    def run(self, tokenized: DatasetDict) -> Trainer:
        """Train and evaluate model, logging to MLflow."""
        model = load_sequence_classifier(
            self.config.model.name,
            num_labels=self.config.model.num_labels,
            id2label=self.config.model.id2label,
            label2id=self.config.model.label2id,
        )
        model = apply_peft(model, self.config.peft)

        args = TrainingArguments(
            output_dir=str(self.config.training.output_dir),
            num_train_epochs=self.config.training.epochs,
            learning_rate=self.config.training.learning_rate,
            per_device_train_batch_size=self.config.training.train_batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.scheduler,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            bf16=self.config.training.bf16,
            fp16=self.config.training.fp16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=3,
            report_to="none",
            eval_strategy="steps",
            save_strategy="steps",
            seed=self.config.training.seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation", tokenized.get("test")),
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(self.config.training.early_stopping_patience)],
            compute_metrics=self._compute_metrics,
        )

        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run(run_name=self.config.run_name):
            mlflow.log_params(
                {
                    "model_name": resolve_model_name(self.config.model.name),
                    "task": self.config.model.task,
                    "epochs": self.config.training.epochs,
                    "lr": self.config.training.learning_rate,
                    "batch_size": self.config.training.train_batch_size,
                }
            )
            checkpoint_path = self._latest_checkpoint(Path(self.config.training.output_dir))
            trainer.train(resume_from_checkpoint=str(checkpoint_path) if checkpoint_path else None)
            metrics = trainer.evaluate(tokenized.get("test", tokenized["validation"]))
            mlflow.log_metrics({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
            checkpoint_dir = Path(self.config.training.output_dir) / "best"
            trainer.save_model(checkpoint_dir)
            mlflow.log_artifacts(checkpoint_dir, artifact_path="model")

        return trainer

    def _compute_metrics(self, eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        probabilities = self._safe_softmax(logits)
        artifacts = classification_metrics(
            y_true=np.asarray(labels),
            y_pred=np.asarray(predictions),
            y_proba=np.asarray(probabilities),
            average=self.config.evaluation.average,
        )
        return artifacts.metrics

    @staticmethod
    def _safe_softmax(logits: np.ndarray) -> np.ndarray:
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shift)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    @staticmethod
    def _latest_checkpoint(output_dir: Path) -> Path | None:
        if not output_dir.exists():
            return None
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        return checkpoints[-1] if checkpoints else None
