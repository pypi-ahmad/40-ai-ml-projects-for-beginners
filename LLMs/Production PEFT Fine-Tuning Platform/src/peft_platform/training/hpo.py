"""Optuna hyperparameter optimization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class HpoResult:
    best_value: float
    best_params: dict[str, float | int | str]
    n_trials: int


def run_optuna(
    objective_fn: Callable[["optuna.trial.Trial"], float],
    n_trials: int = 20,
    study_name: str = "peft_hpo",
) -> HpoResult:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna package not available") from exc

    study = optuna.create_study(direction="minimize", study_name=study_name)
    study.optimize(objective_fn, n_trials=n_trials)
    return HpoResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=n_trials,
    )
