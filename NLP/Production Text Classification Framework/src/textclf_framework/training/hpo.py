"""Optuna hyperparameter optimization for text classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import optuna
from datasets import DatasetDict
from loguru import logger

from textclf_framework.training.engine import TextClassificationTrainer, TrainingRunConfig


@dataclass(slots=True)
class HPOConfig:
    n_trials: int = 20
    timeout_sec: int = 3600


class OptunaTuner:
    """Runs Optuna studies against the training engine."""

    def __init__(
        self,
        dataset_dict: DatasetDict,
        num_labels: int,
        label_names: list[str],
        model_name: str,
        strategy: str,
        output_root: Path,
    ) -> None:
        self.dataset_dict = dataset_dict
        self.num_labels = num_labels
        self.label_names = label_names
        self.model_name = model_name
        self.strategy = strategy
        self.output_root = output_root

    def _objective(self, trial: optuna.Trial) -> float:
        run_config = TrainingRunConfig(
            output_dir=self.output_root / f"trial_{trial.number}",
            model_name=self.model_name,
            strategy=self.strategy,
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 0.1, log=True),
            epochs=trial.suggest_int("epochs", 1, 4),
            train_batch_size=trial.suggest_categorical("train_batch_size", [8, 16, 32]),
            eval_batch_size=trial.suggest_categorical("eval_batch_size", [16, 32, 64]),
            gradient_accumulation_steps=trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.2),
            max_length=trial.suggest_categorical("max_length", [128, 192, 256, 384]),
            early_stopping_patience=2,
            gradient_checkpointing=True,
            fp16=True,
        )

        result = TextClassificationTrainer(run_config).train(
            dataset_dict=self.dataset_dict,
            num_labels=self.num_labels,
            label_names=self.label_names,
        )
        macro_f1 = result.metrics.get("eval_macro_f1", 0.0)
        return float(macro_f1)

    def optimize(self, config: HPOConfig) -> optuna.Study:
        """Launch HPO study and return study object."""
        study = optuna.create_study(direction="maximize", study_name=f"{self.model_name}_hpo")
        logger.info(
            f"Optuna start model={self.model_name} strategy={self.strategy} trials={config.n_trials}"
        )
        study.optimize(self._objective, n_trials=config.n_trials, timeout=config.timeout_sec)
        return study
