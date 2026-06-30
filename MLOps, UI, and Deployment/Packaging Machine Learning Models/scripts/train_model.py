"""Train and benchmark Iris models with reproducible packaging outputs."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_package import ModelLoader, VersionRegistry, setup_logging
from ml_package.artifact_security import compute_sha256
from ml_package.benchmarking import (
    evaluate_model_candidates,
    export_benchmark_rows,
    run_automl_baselines,
    select_best_model,
)
from ml_package.serialization_benchmark import (
    benchmark_serialization,
    write_serialization_benchmark,
)

RANDOM_SEED = 42

logger = setup_logging("train_model")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    payload = np.hstack([X, y.reshape(-1, 1)]).astype(np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


def _ensure_dirs() -> tuple[Path, Path]:
    outputs_dir = Path("outputs")
    benchmarks_dir = outputs_dir / "benchmarks"
    models_dir = Path("models")
    for directory in (benchmarks_dir, outputs_dir / "figures", models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return benchmarks_dir, models_dir


def _multi_class_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        )
    else:
        metrics["roc_auc_ovr"] = None
    return metrics


def _select_v2_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    non_logistic = [row for row in successful if row["model_name"] != "LogisticRegression"]
    candidates = non_logistic if non_logistic else successful
    return sorted(candidates, key=lambda row: (row["f1_macro"], row["accuracy"]), reverse=True)[0]


def _build_torchscript_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[Any | None, str]:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, "torch_not_installed"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    scripted = torch.jit.script(model.cpu())
    return scripted, f"device_used={device}"


def main() -> None:
    from sklearn.datasets import load_iris

    _set_seed(RANDOM_SEED)
    benchmarks_dir, models_dir = _ensure_dirs()

    logger.info("Loading Iris dataset")
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info("Train size=%s, Test size=%s", X_train.shape[0], X_test.shape[0])

    # Version 1 baseline model (deterministic baseline story for tutorial).
    v1_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    v1_model.fit(X_train, y_train)
    v1_metrics = _multi_class_metrics(v1_model, X_test, y_test)

    benchmark_rows = evaluate_model_candidates(
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=RANDOM_SEED,
    )
    export_benchmark_rows(
        benchmark_rows,
        str(benchmarks_dir / "model_benchmark.csv"),
        str(benchmarks_dir / "model_benchmark.json"),
    )

    v2_row = _select_v2_row(benchmark_rows)
    best_overall = select_best_model(benchmark_rows)
    v2_model = v2_row["_model"]
    v2_metrics = _multi_class_metrics(v2_model, X_test, y_test)

    logger.info(
        "v1 baseline: LogisticRegression | accuracy=%.4f | f1_macro=%.4f",
        v1_metrics["accuracy"],
        v1_metrics["f1_macro"],
    )
    logger.info(
        "v2 candidate: %s | accuracy=%.4f | f1_macro=%.4f",
        v2_row["model_name"],
        v2_metrics["accuracy"],
        v2_metrics["f1_macro"],
    )
    logger.info(
        "Best overall benchmark model: %s | accuracy=%.4f | f1_macro=%.4f",
        best_overall["model_name"],
        best_overall["accuracy"],
        best_overall["f1_macro"],
    )

    with (benchmarks_dir / "version_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "v1": {"model_name": "LogisticRegression", **v1_metrics},
                "v2": {"model_name": v2_row["model_name"], **v2_metrics},
                "best_overall_model": best_overall["model_name"],
            },
            handle,
            indent=2,
        )

    v2_report = classification_report(y_test, v2_model.predict(X_test), target_names=target_names)
    with (benchmarks_dir / "classification_report_v2.txt").open("w", encoding="utf-8") as handle:
        handle.write(v2_report)

    automl_rows = run_automl_baselines(X_train, y_train, X_test, y_test)
    with (benchmarks_dir / "automl_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(automl_rows, handle, indent=2)

    # Persist v1 baseline artifacts.
    v1_pkl_path = models_dir / "iris_model_v1.pkl"
    v1_joblib_path = models_dir / "iris_model_v1.joblib"
    ModelLoader(v1_pkl_path).save(
        v1_model,
        create_manifest=True,
        metadata={"version": "v1", "model_name": "LogisticRegression", "seed": RANDOM_SEED},
    )
    ModelLoader(v1_joblib_path).save(
        v1_model,
        create_manifest=True,
        metadata={"version": "v1", "model_name": "LogisticRegression", "seed": RANDOM_SEED},
    )

    # Persist v2 production candidate artifacts.
    v2_pkl_path = models_dir / "iris_model_v2.pkl"
    v2_joblib_path = models_dir / "iris_model_v2.joblib"
    ModelLoader(v2_pkl_path).save(
        v2_model,
        create_manifest=True,
        metadata={"version": "v2", "model_name": v2_row["model_name"], "seed": RANDOM_SEED},
    )
    ModelLoader(v2_joblib_path).save(
        v2_model,
        create_manifest=True,
        metadata={"version": "v2", "model_name": v2_row["model_name"], "seed": RANDOM_SEED},
    )

    # Active default artifact expected by API/CLI.
    active_pkl_path = models_dir / "iris_model.pkl"
    ModelLoader(active_pkl_path).save(
        v2_model,
        create_manifest=True,
        metadata={"version": "v2", "model_name": v2_row["model_name"], "seed": RANDOM_SEED},
    )
    ModelLoader(models_dir / "iris_model.joblib").save(
        v2_model,
        create_manifest=True,
        metadata={"version": "v2", "model_name": v2_row["model_name"], "seed": RANDOM_SEED},
    )

    onnx_loader = ModelLoader(models_dir / "iris_model.onnx")
    try:
        onnx_loader.save(
            v2_model,
            create_manifest=True,
            metadata={"version": "v2", "model_name": v2_row["model_name"], "seed": RANDOM_SEED},
        )
    except Exception as exc:
        logger.warning("ONNX export skipped: %s", exc)

    torchscript_model, torch_build_info = _build_torchscript_model(
        X_train,
        y_train,
        seed=RANDOM_SEED,
    )
    logger.info("TorchScript build info: %s", torch_build_info)

    serialization_rows = benchmark_serialization(
        v2_model,
        artifact_stem="iris_model_benchmark",
        output_dir=models_dir,
        torchscript_model=torchscript_model,
    )
    write_serialization_benchmark(
        serialization_rows,
        benchmarks_dir / "serialization_benchmark.json",
    )

    dataset_fingerprint = _dataset_fingerprint(X, y)
    active_digest = compute_sha256(active_pkl_path)
    v1_digest = compute_sha256(v1_pkl_path)

    registry_path = models_dir / "registry.json"
    if registry_path.exists():
        registry_path.unlink()
    registry = VersionRegistry(str(registry_path))

    registry.register(
        "v1",
        str(v1_pkl_path),
        metrics={
            **v1_metrics,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": len(target_names),
            "seed": RANDOM_SEED,
        },
        description="Baseline logistic regression package artifact",
        author="train_model.py",
        artifact_sha256=v1_digest,
        dataset_fingerprint=dataset_fingerprint,
        tags=["iris", "baseline", "v1"],
        allow_overwrite=True,
    )
    registry.activate("v1")

    registry.register(
        "v2",
        str(active_pkl_path),
        metrics={
            "model_name": v2_row["model_name"],
            **v2_metrics,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": len(target_names),
            "seed": RANDOM_SEED,
        },
        description="Benchmark-selected production candidate",
        author="train_model.py",
        artifact_sha256=active_digest,
        dataset_fingerprint=dataset_fingerprint,
        parent_version="v1",
        tags=["iris", "benchmark", "production_candidate", "v2"],
        allow_overwrite=True,
    )
    registry.activate("v2")

    background_path = models_dir / "background_data.npy"
    np.save(str(background_path), X_train[: min(50, len(X_train))])

    logger.info("Version registry initialized with v1 -> v2 activation flow")
    logger.info("Saved benchmark outputs to %s", benchmarks_dir)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
