"""Training orchestration with Hugging Face Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect

import numpy as np
import torch
from datasets import DatasetDict
from loguru import logger
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from textclf_framework.models.factory import load_model_bundle
from textclf_framework.tokenization.pipeline import TokenizationConfig, TokenizationPipeline
from textclf_framework.training.metrics import compute_metrics_for_trainer


@dataclass(slots=True)
class TrainingRunConfig:
    output_dir: Path
    model_name: str
    strategy: str
    learning_rate: float
    weight_decay: float
    epochs: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_length: int
    early_stopping_patience: int = 2
    gradient_checkpointing: bool = False
    fp16: bool = False
    resume_from_checkpoint: str | None = None


@dataclass(slots=True)
class TrainingResult:
    model_dir: Path
    metrics: dict[str, float]
    trainable_params: int
    total_params: int
    strategy_used: str


class TextClassificationTrainer:
    """End-to-end trainer wrapper for model training/evaluation."""

    def __init__(self, config: TrainingRunConfig) -> None:
        self.config = config

    def _training_args(self) -> TrainingArguments:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        kwargs = {
            "output_dir": str(self.config.output_dir),
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "per_device_train_batch_size": self.config.train_batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "num_train_epochs": self.config.epochs,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_ratio": self.config.warmup_ratio,
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 25,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_macro_f1",
            "greater_is_better": True,
            "fp16": self.config.fp16 and torch.cuda.is_available() and not use_bf16,
            "bf16": use_bf16,
            "report_to": ["tensorboard"],
            "save_total_limit": 2,
            "dataloader_num_workers": 2,
            "remove_unused_columns": False,
        }

        arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        if "evaluation_strategy" in arg_names:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            kwargs["eval_strategy"] = "epoch"

        return TrainingArguments(**kwargs)

    def train(
        self,
        dataset_dict: DatasetDict,
        num_labels: int,
        label_names: list[str],
    ) -> TrainingResult:
        """Train and evaluate a text classification model."""
        bundle = load_model_bundle(
            model_name=self.config.model_name,
            num_labels=num_labels,
            strategy=self.config.strategy,
            gradient_checkpointing=self.config.gradient_checkpointing,
            use_4bit=self.config.strategy == "qlora",
        )

        tokenizer_pipeline = TokenizationPipeline(
            TokenizationConfig(
                model_name=bundle.tokenizer.name_or_path,
                max_length=self.config.max_length,
                truncation=True,
                padding=False,
            )
        )
        tokenized = tokenizer_pipeline.tokenize_dataset(dataset_dict)
        tokenized = tokenized.rename_column("label", "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        trainer_kwargs = {
            "model": bundle.model,
            "args": self._training_args(),
            "train_dataset": tokenized["train"],
            "eval_dataset": tokenized["validation"],
            "data_collator": tokenizer_pipeline.collator(),
            "compute_metrics": compute_metrics_for_trainer,
            "callbacks": [EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        }
        trainer_arg_names = set(inspect.signature(Trainer.__init__).parameters.keys())
        if "tokenizer" in trainer_arg_names:
            trainer_kwargs["tokenizer"] = bundle.tokenizer
        else:
            trainer_kwargs["processing_class"] = bundle.tokenizer

        trainer = Trainer(**trainer_kwargs)

        logger.info(
            f"Training start model={self.config.model_name} strategy={self.config.strategy} labels={len(label_names)}"
        )
        trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

        eval_metrics = trainer.evaluate(tokenized["test"])
        model_dir = self.config.output_dir / "best_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_dir)
        bundle.tokenizer.save_pretrained(model_dir)

        sanitized_metrics = {
            key: float(value) if isinstance(value, (int, float, np.floating)) else float("nan")
            for key, value in eval_metrics.items()
        }

        return TrainingResult(
            model_dir=model_dir,
            metrics=sanitized_metrics,
            trainable_params=bundle.trainable_params,
            total_params=bundle.total_params,
            strategy_used=bundle.strategy,
        )
