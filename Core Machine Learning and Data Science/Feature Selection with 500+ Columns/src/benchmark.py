"""
benchmark.py
------------
Benchmarking utilities for feature selection project.

Provides:
  - LazyPredictClassifier wrapper for rapid baselines
  - PyCaret comparison wrapper
  - FLAML hyperparameter optimization wrapper
  - Manual model training/evaluation with multiple metrics
  - Before/after feature selection comparison

Design:
  Each tool is benchmarked independently with clear output.
  Results are returned as DataFrames for easy comparison.
  Timing and memory tracking included.

Inputs:
  - X_train, X_test, y_train, y_test
  - List of model classes

Outputs:
  - DataFrame of metrics per model
  - Training/inference times
  - Comparison tables
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ------------------------------------------------------------------
# Common metric computation
# ------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute a standard set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities (for ROC-AUC).

    Returns
    -------
    dict of {metric_name: value}
    """
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "roc_auc": 0.0,
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                y_proba_binary = (
                    y_proba[:, 1]
                    if getattr(y_proba, "ndim", 1) == 2 and y_proba.shape[1] >= 2
                    else y_proba
                )
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_binary))
            elif getattr(y_proba, "ndim", 1) == 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            metrics["roc_auc"] = 0.0
    return metrics


# ------------------------------------------------------------------
# LazyPredict Wrapper
# ------------------------------------------------------------------
def lazy_predict_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: int = 0,
    classifiers: str = "all",
) -> pd.DataFrame:
    """
    Run LazyPredict to get rapid baseline model rankings.

    LazyPredict trains many classifiers quickly and returns a
    comparison DataFrame. This is useful for understanding which
    model families work well before doing any hyperparameter tuning.

    How it works:
      1. Creates a LazyClassifier instance.
      2. Fits on training data.
      3. Returns a DataFrame with Accuracy, F1, ROC-AUC, etc.

    Limitations:
      - No hyperparameter tuning (default params only).
      - Some models may fail to converge on large data.
      - Can be slow for 500+ features due to many models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    verbose : int
        LazyPredict verbosity.
    classifiers : str or list
        'all' or list of classifier names.

    Returns
    -------
    pd.DataFrame
        Model comparison table from LazyPredict.
    """
    from lazypredict.Supervised import LazyClassifier

    lazy_clf = LazyClassifier(
        verbose=verbose,
        ignore_warnings=True,
        custom_metric=None,
        predictions=True,
        classifiers=classifiers,
    )
    models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
    return models


# ------------------------------------------------------------------
# PyCaret Wrapper
# ------------------------------------------------------------------
def pycaret_compare(
    data: pd.DataFrame,
    target: str,
    fold: int = 5,
    max_features: float = 0.8,
    test_data: Optional[pd.DataFrame] = None,
    return_holdout: bool = False,
    include_models: Optional[List[str]] = None,
) -> Any:
    """
    Use PyCaret to compare multiple models with cross-validation.

    PyCaret provides an automated ML workflow with preprocessing,
    model comparison, tuning, and ensembling.

    How it works:
      1. PyCaret's setup() handles preprocessing automatically.
      2. compare_models() trains many models with CV.
      3. Returns a ranked comparison table.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset (features + target).
    target : str
        Name of target column.
    fold : int
        Number of CV folds.
    max_features : float
        Fraction of features to sample for some models.

    test_data : pd.DataFrame or None
        Optional external holdout data (must include target column).
        If provided, top model is scored on this holdout via predict_model.
    return_holdout : bool
        If True, return dict with leaderboard + holdout metrics.
    include_models : list[str] or None
        Optional subset of PyCaret model IDs for faster comparisons.

    Returns
    -------
    pd.DataFrame or dict
        Default: leaderboard DataFrame.
        If return_holdout=True:
        {
            "leaderboard": pd.DataFrame,
            "holdout_metrics": dict | None,
            "best_model_name": str,
        }
    """
    try:
        from pycaret.classification import compare_models, predict_model, pull, setup
    except Exception as exc:  # pragma: no cover - depends on local interpreter/runtime
        warnings.warn(
            f"PyCaret is unavailable in this environment: {exc}",
            RuntimeWarning,
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "Model": "PyCaretUnavailable",
                    "Status": "error",
                    "Reason": str(exc),
                }
            ]
        )
        if return_holdout:
            return {
                "leaderboard": leaderboard,
                "holdout_metrics": None,
                "best_model_name": "PyCaretUnavailable",
                "error": str(exc),
            }
        return leaderboard

    setup(
        data=data,
        target=target,
        fold=fold,
        session_id=42,
        verbose=False,
        html=False,
    )
    compare_kwargs = {"n_select": 1, "verbose": False}
    if include_models:
        compare_kwargs["include"] = include_models
    best_model = compare_models(**compare_kwargs)
    leaderboard = pull()  # Get comparison dataframe

    holdout_metrics = None
    best_model_name = str(leaderboard.iloc[0].get("Model", type(best_model).__name__))
    if test_data is not None:
        pred_df = predict_model(best_model, data=test_data, verbose=False)
        y_true = pred_df[target].values
        y_pred = pred_df["prediction_label"].values

        y_proba = None
        proba_cols = [c for c in pred_df.columns if c.startswith("prediction_score_")]
        if proba_cols:
            y_proba = pred_df[proba_cols].values
        elif "prediction_score" in pred_df.columns:
            y_proba = pred_df["prediction_score"].values

        holdout_metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)
        holdout_metrics["model_name"] = best_model_name

    if return_holdout:
        return {
            "leaderboard": leaderboard,
            "holdout_metrics": holdout_metrics,
            "best_model_name": best_model_name,
        }

    return leaderboard


# ------------------------------------------------------------------
# FLAML Wrapper
# ------------------------------------------------------------------
def flaml_optimize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget: int = 60,
    metric: str = "accuracy",
    task: str = "classification",
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Use FLAML to find the best model and hyperparameters.

    FLAML (Fast Lightweight AutoML) efficiently searches over
    model families and hyperparameters using cost-aware search.

    How it works:
      1. Defines the ML task.
      2. FLAML searches over models and hyperparameters.
      3. Returns the best estimator and config.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    time_budget : int
        Seconds to search.
    metric : str
        Optimization metric ('accuracy', 'f1', 'roc_auc').
    task : str
        'classification' or 'regression'.

    Returns
    -------
    tuple
        (best_estimator, config_dict)
    """
    from flaml import AutoML

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=task,
        time_budget=time_budget,
        metric=metric,
        **kwargs,
    )

    # Evaluate on test set
    y_pred = automl.predict(X_test)
    y_proba = automl.predict_proba(X_test) if hasattr(automl, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_proba)

    result = {
        "best_estimator": automl,
        "best_config": automl.best_config,
        "best_loss": automl.best_loss,
        "metrics": metrics,
        "model_history": automl.model_history if hasattr(automl, "model_history") else {},
    }
    return automl, result


# ------------------------------------------------------------------
# Manual Model Training and Evaluation
# ------------------------------------------------------------------
def train_and_evaluate(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Train a single model and return comprehensive metrics.

    Includes training time, inference time, and full classification metrics.

    Parameters
    ----------
    model : BaseEstimator
        sklearn-compatible classifier.
    X_train, y_train : training data
    X_test, y_test : test data
    model_name : str
        Display name for the model.

    Returns
    -------
    dict
        {
            'model_name': str,
            'model': BaseEstimator (fitted),
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'roc_auc': float,
            'train_time': float (seconds),
            'inference_time': float (seconds),
        }
    """
    import tracemalloc

    tracemalloc.start()

    # Training
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, train_peak = tracemalloc.get_traced_memory()

    # Inference
    tracemalloc.reset_peak()
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    _, inference_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Probabilities for ROC-AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["model_name"] = model_name
    metrics["train_time_sec"] = round(train_time, 4)
    metrics["inference_time_sec"] = round(inference_time, 4)
    metrics["train_peak_mem_mb"] = round(train_peak / (1024 * 1024), 3)
    metrics["inference_peak_mem_mb"] = round(inference_peak / (1024 * 1024), 3)
    metrics["model"] = model

    return metrics


# ------------------------------------------------------------------
# Compare Multiple Models
# ------------------------------------------------------------------
def compare_models(
    models: List[Tuple[str, BaseEstimator]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Train multiple models and return a comparison DataFrame.

    Parameters
    ----------
    models : list of (name, estimator) tuples
    X_train, y_train, X_test, y_test : data

    Returns
    -------
    pd.DataFrame
        One row per model with all metrics.
    """
    results = []
    for name, model in models:
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        results.append(metrics)

    df = pd.DataFrame(results)
    cols = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "train_time_sec",
        "inference_time_sec",
        "train_peak_mem_mb",
        "inference_peak_mem_mb",
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].set_index("model_name")


# ------------------------------------------------------------------
# Before/After Feature Selection Comparison
# ------------------------------------------------------------------
def compare_before_after(
    model: BaseEstimator,
    X_train_full: pd.DataFrame,
    X_test_full: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_selected: pd.DataFrame,
    X_test_selected: pd.DataFrame,
    model_name: str = "Model",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare model performance before and after feature selection.

    Parameters
    ----------
    model : BaseEstimator
        Model instance (will be cloned internally).
    X_train_full, X_test_full : Full feature sets.
    y_train, y_test : Targets.
    X_train_selected, X_test_selected : Reduced feature sets.
    model_name : str
        Display name.

    Returns
    -------
    pd.DataFrame
        Rows: 'Before FS', 'After FS'.
        Columns: metrics + improvement.
    """
    from sklearn.base import clone

    # Before
    model_before = clone(model)
    before = train_and_evaluate(
        model_before, X_train_full, y_train, X_test_full, y_test,
        f"{model_name} (Before FS)",
    )

    # After
    model_after = clone(model)
    after = train_and_evaluate(
        model_after, X_train_selected, y_train, X_test_selected, y_test,
        f"{model_name} (After FS)",
    )

    comparison = pd.DataFrame([before, after])
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    change = {}
    for col in metric_cols:
        change[col + "_change"] = comparison[col].iloc[1] - comparison[col].iloc[0]
        change[col + "_pct"] = (
            (comparison[col].iloc[1] - comparison[col].iloc[0])
            / comparison[col].iloc[0]
            * 100
            if comparison[col].iloc[0] != 0
            else 0
        )
    change["train_time_speedup"] = (
        comparison["train_time_sec"].iloc[0] - comparison["train_time_sec"].iloc[1]
    )
    change["inference_time_speedup"] = (
        comparison["inference_time_sec"].iloc[0] - comparison["inference_time_sec"].iloc[1]
    )
    if "train_peak_mem_mb" in comparison.columns:
        change["train_peak_mem_reduction_mb"] = (
            comparison["train_peak_mem_mb"].iloc[0] - comparison["train_peak_mem_mb"].iloc[1]
        )
    if "inference_peak_mem_mb" in comparison.columns:
        change["inference_peak_mem_reduction_mb"] = (
            comparison["inference_peak_mem_mb"].iloc[0]
            - comparison["inference_peak_mem_mb"].iloc[1]
        )

    return comparison, change
