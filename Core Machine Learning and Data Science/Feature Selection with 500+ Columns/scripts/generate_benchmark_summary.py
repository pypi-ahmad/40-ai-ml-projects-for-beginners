"""Generate a reproducible benchmark summary on ISOLET.

Outputs:
  - outputs/metrics/benchmark_summary.csv
  - outputs/metrics/benchmark_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _append_compare_rows(
    rows: list[dict[str, Any]],
    tool_name: str,
    comparison_df: pd.DataFrame,
    feature_counts: dict[str, int],
) -> None:
    for _, row in comparison_df.iterrows():
        condition = "After FS" if "After FS" in str(row.get("model_name", "")) else "Before FS"
        rows.append(
            {
                "tool": tool_name,
                "model_name": row.get("model_name"),
                "condition": condition,
                "accuracy": _safe_float(row.get("accuracy")),
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "roc_auc": _safe_float(row.get("roc_auc")),
                "train_time_sec": _safe_float(row.get("train_time_sec")),
                "inference_time_sec": _safe_float(row.get("inference_time_sec")),
                "train_peak_mem_mb": _safe_float(row.get("train_peak_mem_mb")),
                "inference_peak_mem_mb": _safe_float(row.get("inference_peak_mem_mb")),
                "n_features": feature_counts[condition],
                "error": None,
            }
        )


def _manual_models(random_state: int) -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = [
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1200,
                n_jobs=-1,
                random_state=random_state,
                multi_class="ovr",
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=220,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=220,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        ("LinearSVM", LinearSVC(C=1.0, random_state=random_state)),
        ("KNN", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ]

    return models


def main() -> int:
    from src.benchmark import (
        compare_before_after,
        flaml_optimize,
        lazy_predict_baseline,
        pycaret_compare,
    )
    from src.data_loader import load_isolet_dataset
    from src.feature_selector import FeatureSelector

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    random_state = 42

    X, y, metadata = load_isolet_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )
    fs_size = min(1500, len(X_train_fs))
    X_train_fs_small, _, y_train_fs_small, _ = train_test_split(
        X_train_fs,
        y_train_fs,
        train_size=fs_size,
        random_state=random_state,
        stratify=y_train_fs,
    )

    selector = FeatureSelector(random_state=random_state)
    LOGGER.info("Running feature-selection funnel to derive reduced feature subset")
    selected = selector.pipeline(
        X_train_fs_small,
        y_train_fs_small,
        X_val=X_val_fs,
        y_val=y_val_fs,
        var_threshold=0.0,
        corr_threshold=0.90,
        corr_method="spearman",
        rfe_feat=120,
        rfe_step=15,
        rfe_use_cv=False,
        rfe_min_features=30,
        l1_C=1.0,
        mi_k=100,
        shap_k=60,
        verbose=False,
    )

    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]
    feature_counts = {"Before FS": int(X_train.shape[1]), "After FS": int(X_train_sel.shape[1])}
    bench_train_size = min(2200, len(X_train))
    X_train_bench, _, y_train_bench, _ = train_test_split(
        X_train,
        y_train,
        train_size=bench_train_size,
        random_state=random_state,
        stratify=y_train,
    )
    X_train_sel_bench = X_train_bench[selected]

    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    for name, model in _manual_models(random_state):
        LOGGER.info("Benchmarking manual model: %s", name)
        try:
            comparison, _ = compare_before_after(
                model=model,
                X_train_full=X_train_bench,
                X_test_full=X_test,
                y_train=y_train_bench,
                y_test=y_test,
                X_train_selected=X_train_sel_bench,
                X_test_selected=X_test_sel,
                model_name=name,
            )
            _append_compare_rows(rows, "ManualModel", comparison, feature_counts)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            notes.append(f"Manual model {name} failed: {exc}")

    for condition, xtr, xte in [
        ("Before FS", X_train_bench, X_test),
        ("After FS", X_train_sel_bench, X_test_sel),
    ]:
        LOGGER.info("Running LazyPredict (%s)", condition)
        try:
            lazy_df = lazy_predict_baseline(
                xtr,
                y_train_bench,
                xte,
                y_test,
                verbose=0,
                classifiers=[
                    RandomForestClassifier,
                    ExtraTreesClassifier,
                    LogisticRegression,
                    LinearSVC,
                    KNeighborsClassifier,
                ],
            )
            if not lazy_df.empty:
                top = lazy_df.iloc[0]
                rows.append(
                    {
                        "tool": "LazyPredict(top-1)",
                        "model_name": str(lazy_df.index[0]),
                        "condition": condition,
                        "accuracy": _safe_float(top.get("Accuracy")),
                        "precision": None,
                        "recall": None,
                        "f1": _safe_float(top.get("F1 Score")),
                        "roc_auc": _safe_float(top.get("ROC AUC")),
                        "train_time_sec": _safe_float(top.get("Time Taken")),
                        "inference_time_sec": None,
                        "train_peak_mem_mb": None,
                        "inference_peak_mem_mb": None,
                        "n_features": feature_counts[condition],
                        "error": None,
                    }
                )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            notes.append(f"LazyPredict {condition} failed: {exc}")

    for condition, xtr, xte in [
        ("Before FS", X_train_bench, X_test),
        ("After FS", X_train_sel_bench, X_test_sel),
    ]:
        LOGGER.info("Running PyCaret compare (%s)", condition)
        data = xtr.copy()
        data["target"] = y_train_bench.values
        holdout = xte.copy()
        holdout["target"] = y_test.values
        try:
            pycaret_result = pycaret_compare(
                data=data,
                target="target",
                fold=2,
                test_data=holdout,
                return_holdout=True,
                include_models=["lr", "rf", "et", "knn", "svm"],
            )
            leaderboard = pycaret_result["leaderboard"]
            holdout_metrics = pycaret_result["holdout_metrics"]
            if holdout_metrics is not None:
                rows.append(
                    {
                        "tool": "PyCaret(top-1)",
                        "model_name": str(pycaret_result.get("best_model_name", "PyCaretBestModel")),
                        "condition": condition,
                        "accuracy": _safe_float(holdout_metrics.get("accuracy")),
                        "precision": _safe_float(holdout_metrics.get("precision")),
                        "recall": _safe_float(holdout_metrics.get("recall")),
                        "f1": _safe_float(holdout_metrics.get("f1")),
                        "roc_auc": _safe_float(holdout_metrics.get("roc_auc")),
                        "train_time_sec": _safe_float(leaderboard.iloc[0].get("TT (Sec)"))
                        if not leaderboard.empty
                        else None,
                        "inference_time_sec": None,
                        "train_peak_mem_mb": None,
                        "inference_peak_mem_mb": None,
                        "n_features": feature_counts[condition],
                        "error": None,
                    }
                )
            else:
                rows.append(
                    {
                        "tool": "PyCaret(top-1)",
                        "model_name": str(pycaret_result.get("best_model_name", "PyCaretUnavailable")),
                        "condition": condition,
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "roc_auc": None,
                        "train_time_sec": _safe_float(leaderboard.iloc[0].get("TT (Sec)"))
                        if not leaderboard.empty
                        else None,
                        "inference_time_sec": None,
                        "train_peak_mem_mb": None,
                        "inference_peak_mem_mb": None,
                        "n_features": feature_counts[condition],
                        "error": pycaret_result.get(
                            "error",
                            "PyCaret holdout metrics unavailable in current runtime.",
                        ),
                    }
                )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            notes.append(f"PyCaret {condition} failed: {exc}")

    for condition, xtr, xte in [
        ("Before FS", X_train_bench, X_test),
        ("After FS", X_train_sel_bench, X_test_sel),
    ]:
        LOGGER.info("Running FLAML optimization (%s)", condition)
        try:
            t0 = perf_counter()
            automl, result = flaml_optimize(
                xtr,
                y_train_bench,
                xte,
                y_test,
                time_budget=20,
                metric="accuracy",
                task="classification",
                estimator_list=["rf", "extra_tree", "sgd", "lrl1"],
                log_training_metric=False,
            )
            total_time = perf_counter() - t0
            rows.append(
                {
                    "tool": "FLAML",
                    "model_name": str(automl.best_estimator),
                    "condition": condition,
                    "accuracy": _safe_float(result["metrics"].get("accuracy")),
                    "precision": _safe_float(result["metrics"].get("precision")),
                    "recall": _safe_float(result["metrics"].get("recall")),
                    "f1": _safe_float(result["metrics"].get("f1")),
                    "roc_auc": _safe_float(result["metrics"].get("roc_auc")),
                    "train_time_sec": round(total_time, 4),
                    "inference_time_sec": None,
                    "train_peak_mem_mb": None,
                    "inference_peak_mem_mb": None,
                    "n_features": feature_counts[condition],
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            notes.append(f"FLAML {condition} failed: {exc}")

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["tool", "model_name", "condition"], inplace=True, ignore_index=True)
    out_df.to_csv(METRICS_DIR / "benchmark_summary.csv", index=False)

    payload = {
        "dataset": metadata,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "original_features": feature_counts["Before FS"],
        "selected_features": feature_counts["After FS"],
        "rows": rows,
        "notes": notes,
        "random_state": random_state,
    }
    with (METRICS_DIR / "benchmark_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    LOGGER.info("Wrote benchmark summary to %s", METRICS_DIR)
    if notes:
        for note in notes:
            LOGGER.warning("%s", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
