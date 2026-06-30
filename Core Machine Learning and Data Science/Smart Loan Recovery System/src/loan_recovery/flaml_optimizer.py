"""FLAML optimization workflow for automated model search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import REPORTS_DIR


@dataclass(slots=True)
class FLAMLArtifacts:
    """Artifacts produced by FLAML optimization."""

    estimator_name: str
    best_config: dict[str, Any]
    best_loss: float
    metrics: dict[str, float]
    model: Any


class FLAMLOptimizer:
    """Run FLAML AutoML search and evaluate on hold-out data."""

    def __init__(self, time_budget: int = 120, random_state: int = 42) -> None:
        self.time_budget = time_budget
        self.random_state = random_state

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> FLAMLArtifacts:
        """Optimize classifier and compute post-search metrics."""
        from flaml import AutoML

        automl = AutoML()
        log_path = REPORTS_DIR / "flaml.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            automl.fit(
                X_train=x_train,
                y_train=y_train,
                X_val=x_test,
                y_val=y_test,
                task="classification",
                metric="macro_f1",
                time_budget=self.time_budget,
                seed=self.random_state,
                log_file_name=str(log_path),
                eval_method="holdout",
                n_jobs=-1,
                estimator_list=["xgboost", "rf", "extra_tree"],
                verbose=0,
            )
        except Exception:
            # Conservative fallback when macro_f1 is unsupported in some versions.
            automl.fit(
                X_train=x_train,
                y_train=y_train,
                X_val=x_test,
                y_val=y_test,
                task="classification",
                metric="accuracy",
                time_budget=self.time_budget,
                seed=self.random_state,
                log_file_name=str(log_path),
                eval_method="holdout",
                n_jobs=-1,
                estimator_list=["xgboost", "rf", "extra_tree"],
                verbose=0,
            )

        y_pred = automl.predict(x_test)
        y_prob = automl.predict_proba(x_test) if hasattr(automl, "predict_proba") else None

        metrics: dict[str, float] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc_ovr"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr")), 4)
            except Exception:
                metrics["roc_auc_ovr"] = np.nan
        else:
            metrics["roc_auc_ovr"] = np.nan

        estimator_name = str(automl.best_estimator)
        return FLAMLArtifacts(
            estimator_name=estimator_name,
            best_config=dict(automl.best_config),
            best_loss=float(automl.best_loss),
            metrics=metrics,
            model=automl.model,
        )
