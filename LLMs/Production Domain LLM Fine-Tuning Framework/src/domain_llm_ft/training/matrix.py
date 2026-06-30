"""Training matrix orchestration utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.training.pipeline import run_training


PUBLIC_DATASET_MATRIX = [
    "ag_news",
    "dbpedia_14",
    "imdb",
    "emotion",
    "financial_phrasebank",
    "trec",
]

MODEL_MATRIX = [
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


def run_public_matrix(config: ExperimentConfig, execute: bool = False) -> pd.DataFrame:
    """Prepare or execute matrix runs across public datasets and model list."""
    rows: list[dict[str, str]] = []

    for dataset_name in PUBLIC_DATASET_MATRIX:
        for model_name in MODEL_MATRIX:
            run_cfg = deepcopy(config)
            run_cfg.dataset.name = dataset_name
            run_cfg.dataset.source = "hf"
            run_cfg.model.name = model_name
            run_cfg.run_name = f"{dataset_name}_{model_name}"
            run_cfg.training.output_dir = Path(config.training.output_dir) / dataset_name / model_name

            status = "planned"
            if execute:
                run_training(run_cfg)
                status = "completed"

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "run_name": run_cfg.run_name,
                    "status": status,
                }
            )

    return pd.DataFrame(rows)
