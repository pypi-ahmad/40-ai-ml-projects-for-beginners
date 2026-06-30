"""Model benchmarking utilities for reproducible model selection."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _base_candidates(random_state: int) -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def _optional_candidates(random_state: int) -> dict[str, Any]:
    candidates: dict[str, Any] = {}

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            random_state=random_state,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="mlogloss",
            verbosity=0,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBM"] = LGBMClassifier(
            random_state=random_state,
            n_estimators=200,
            learning_rate=0.05,
            verbosity=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        candidates["CatBoost"] = CatBoostClassifier(
            random_state=random_state,
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=False,
        )
    except Exception:
        pass

    return candidates


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    candidates = _base_candidates(random_state)
    candidates.update(_optional_candidates(random_state))
    return candidates


def _score_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | None]:
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    roc_auc: float | None = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            roc_auc = float(
                roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr": roc_auc,
    }


def evaluate_model_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Fit and evaluate candidate models. Includes optional libraries when installed."""
    rows: list[dict[str, Any]] = []

    for model_name, estimator in build_candidate_models(random_state).items():
        model = clone(estimator)
        fit_start = time.perf_counter()
        try:
            model.fit(X_train, y_train)
            fit_time_ms = (time.perf_counter() - fit_start) * 1000

            pred_start = time.perf_counter()
            metrics = _score_model(model, X_test, y_test)
            predict_time_ms = (time.perf_counter() - pred_start) * 1000

            row = {
                "model_name": model_name,
                "fit_time_ms": round(fit_time_ms, 3),
                "predict_time_ms": round(predict_time_ms, 3),
                "supports_proba": hasattr(model, "predict_proba"),
                "status": "ok",
                "error": None,
                "_model": model,
            }
            row.update(metrics)
            rows.append(row)
        except Exception as exc:  # pragma: no cover - defensive path
            rows.append(
                {
                    "model_name": model_name,
                    "fit_time_ms": None,
                    "predict_time_ms": None,
                    "supports_proba": hasattr(model, "predict_proba"),
                    "status": "error",
                    "error": str(exc),
                    "accuracy": None,
                    "precision_macro": None,
                    "recall_macro": None,
                    "f1_macro": None,
                    "roc_auc_ovr": None,
                    "_model": None,
                }
            )

    return rows


def run_automl_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    """Run optional AutoML baselines. Returns skip reasons when packages unavailable."""
    results: list[dict[str, Any]] = []

    try:
        from lazypredict.Supervised import LazyClassifier

        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
        top = models_df.sort_values("F1 Score", ascending=False).head(1)
        results.append(
            {
                "framework": "LazyPredict",
                "status": "ok",
                "best_model": top.index[0],
                "metric_name": "F1 Score",
                "metric_value": float(top.iloc[0]["F1 Score"]),
                "notes": "Auto baseline from LazyPredict candidate search",
            }
        )
    except Exception as exc:
        results.append(
            {
                "framework": "LazyPredict",
                "status": "skipped",
                "best_model": None,
                "metric_name": None,
                "metric_value": None,
                "notes": str(exc),
            }
        )

    try:
        from flaml import AutoML

        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            time_budget=15,
            metric="accuracy",
            verbose=0,
        )
        test_score = float((automl.predict(X_test) == y_test).mean())
        results.append(
            {
                "framework": "FLAML",
                "status": "ok",
                "best_model": automl.best_estimator,
                "metric_name": "accuracy",
                "metric_value": test_score,
                "notes": "Time-budgeted AutoML search",
            }
        )
    except Exception as exc:
        results.append(
            {
                "framework": "FLAML",
                "status": "skipped",
                "best_model": None,
                "metric_name": None,
                "metric_value": None,
                "notes": str(exc),
            }
        )

    try:
        from pycaret.classification import create_model, pull, setup

        data = pd.DataFrame(X_train, columns=[f"feature_{idx}" for idx in range(X_train.shape[1])])
        data["target"] = y_train
        setup(
            data=data,
            target="target",
            session_id=42,
            html=False,
            verbose=False,
            fold=2,
            n_jobs=1,
        )
        # Lightweight PyCaret baseline for predictable tutorial runtime.
        best = create_model("lr")
        table = pull()
        metric = float(table.iloc[0]["F1"])
        results.append(
            {
                "framework": "PyCaret",
                "status": "ok",
                "best_model": type(best).__name__,
                "metric_name": "F1",
                "metric_value": metric,
                "notes": "PyCaret logistic-regression baseline",
            }
        )
    except Exception as exc:
        results.append(
            {
                "framework": "PyCaret",
                "status": "skipped",
                "best_model": None,
                "metric_name": None,
                "metric_value": None,
                "notes": str(exc),
            }
        )

    return results


def select_best_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select best successful model by macro-F1 then accuracy."""
    candidates = [row for row in rows if row["status"] == "ok"]
    if not candidates:
        raise RuntimeError("No successful model candidates available.")

    return sorted(
        candidates,
        key=lambda row: (row["f1_macro"], row["accuracy"]),
        reverse=True,
    )[0]


def export_benchmark_rows(
    rows: list[dict[str, Any]],
    csv_path: str,
    json_path: str,
) -> None:
    """Persist benchmark table as CSV and JSON without in-memory model objects."""
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        cleaned_rows.append({key: value for key, value in row.items() if key != "_model"})

    frame = pd.DataFrame(cleaned_rows)
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)
