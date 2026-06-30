"""Optuna hyperparameter optimization runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import optuna
from loguru import logger

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.data.loader import DatasetLoader
from domain_llm_ft.tokenization.factory import TokenizationPipeline
from domain_llm_ft.training.trainer_engine import TrainerEngine


class OptunaRunner:
    """Run hyperparameter optimization over training pipeline."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, storage_path: Path | None = None) -> optuna.Study:
        """Execute Optuna study and persist summary."""
        storage = f"sqlite:///{storage_path}" if storage_path else None
        study = optuna.create_study(
            direction=self.config.hpo.direction,
            study_name=f"{self.config.experiment_name}_hpo",
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=self.config.hpo.trials,
            timeout=self.config.hpo.timeout_minutes * 60,
        )
        logger.info("Best trial: {}", study.best_trial.params)
        return study

    def _objective(self, trial: optuna.Trial) -> float:
        cfg = deepcopy(self.config)
        cfg.training.learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
        cfg.training.train_batch_size = trial.suggest_categorical("train_batch_size", [4, 8, 16])
        cfg.training.weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
        cfg.tokenizer.max_length = trial.suggest_categorical("max_length", [128, 256, 384, 512])
        cfg.training.epochs = trial.suggest_int("epochs", 1, 4)

        dataset = DatasetLoader(cfg.dataset).load()
        tok = TokenizationPipeline(cfg.tokenizer)
        tokenized = tok.tokenize_dataset(dataset, cfg.dataset.text_column, cfg.dataset.label_column)
        trainer = TrainerEngine(cfg, tok.tokenizer, tok.build_collator())
        run = trainer.run(tokenized)
        metrics = run.evaluate(tokenized.get("test", tokenized.get("validation")))
        score = float(metrics.get(self.config.hpo.metric_name, metrics.get("eval_f1", 0.0)))
        return score
