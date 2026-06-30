"""Top-level pipeline orchestration."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.data.eda import dataset_statistics, render_class_distribution, render_length_distribution
from domain_llm_ft.data.loader import DatasetLoader
from domain_llm_ft.data.preprocess import build_label_mapping
from domain_llm_ft.tokenization.factory import TokenizationPipeline
from domain_llm_ft.training.accelerate_engine import AccelerateEngine
from domain_llm_ft.training.trainer_engine import TrainerEngine
from domain_llm_ft.utils.io import write_json


def prepare_dataset(config: ExperimentConfig) -> tuple:
    """Prepare datasets and tokenization assets."""
    dataset = DatasetLoader(config.dataset).load()

    id2label, label2id = build_label_mapping(dataset["train"], config.dataset.label_column)
    config.model.id2label = id2label
    config.model.label2id = label2id
    config.model.num_labels = len(id2label)

    output_dir = Path(config.paths.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = pd.DataFrame(dataset["train"])
    stats = dataset_statistics(train_frame, config.dataset.text_column, config.dataset.label_column)
    write_json(output_dir / "reports" / "dataset_stats.json", stats)

    render_class_distribution(
        train_frame,
        config.dataset.label_column,
        output_dir / "figures" / "class_distribution.png",
    )
    render_length_distribution(
        train_frame,
        config.dataset.text_column,
        output_dir / "figures" / "length_distribution.png",
    )

    tokenization = TokenizationPipeline(config.tokenizer)
    tokenized = tokenization.tokenize_dataset(
        dataset,
        text_column=config.dataset.text_column,
        label_column=config.dataset.label_column,
    )
    return dataset, tokenization, tokenized


def run_training(config: ExperimentConfig) -> dict[str, str]:
    """Run full training pipeline and return artifact paths."""
    _dataset, tokenization, tokenized = prepare_dataset(config)
    output_dir = Path(config.paths.artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{config.paths.mlruns_dir}")
    if config.training.engine == "trainer":
        trainer = TrainerEngine(config, tokenization.tokenizer, tokenization.build_collator())
        trainer.run(tokenized)
    else:
        accelerator = AccelerateEngine(config)
        result = accelerator.run(tokenized, tokenization.build_collator())
        write_json(output_dir / "reports" / "accelerate_result.json", {"train_loss": result.train_loss})

    return {
        "artifacts_dir": str(output_dir.resolve()),
        "checkpoints": str(Path(config.training.output_dir).resolve()),
    }
