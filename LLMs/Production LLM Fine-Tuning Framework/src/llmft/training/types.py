"""Training output types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingReport:
    """Training run summary."""

    run_id: str
    model_alias: str
    model_id: str
    peft_method: str
    train_samples: int
    validation_samples: int
    steps: int
    train_loss: float
    eval_loss: float
    checkpoints_dir: Path
    used_real_stack: bool


@dataclass(slots=True)
class HPOReport:
    """Hyperparameter optimization summary."""

    run_id: str
    best_params: dict[str, float | int]
    best_score: float
    trials: int
    backend: str
    report_path: Path
