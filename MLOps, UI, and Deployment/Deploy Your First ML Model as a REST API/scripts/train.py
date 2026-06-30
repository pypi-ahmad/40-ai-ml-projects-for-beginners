"""End-to-end training script for Project #9.

Usage:
    uv run python scripts/train.py --profile quick
    uv run python scripts/train.py --profile full --flaml-seconds 120
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import (
    BenchmarkResult,
    run_flaml_benchmark,
    run_lazypredict_benchmark,
    run_pycaret_benchmark,
)
from src.constants import (
    ARTIFACTS_DIR,
    BENCHMARKS_DIR,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
)
from src.data import load_california_housing, split_dataset
from src.serialization import export_onnx_if_available, save_metadata, save_model
from src.training import build_metadata, dump_training_summary, train_best_model
from src.visualization import plot_metric_radar, plot_model_ranking

# Reproducibility seed used across numpy/model libraries.
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def _save_benchmark_result(result: BenchmarkResult) -> dict[str, str]:
    """Persist benchmark output and return status summary row."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = result.tool.lower().replace(" ", "_")
    out_path = BENCHMARKS_DIR / f"{safe_name}_benchmark.csv"

    if not result.table.empty:
        result.table.to_csv(out_path, index=False)

    return {
        "tool": result.tool,
        "status": result.status,
        "rows": str(len(result.table)),
        "notes": result.notes,
        "path": str(out_path if not result.table.empty else ""),
    }


def _write_benchmark_evidence(profile: str, benchmark_status_rows: list[dict[str, str]]) -> None:
    """Create narrative benchmark evidence markdown for portfolio/readers."""
    evidence_path = REPORTS_DIR / "benchmark_evidence.md"
    with evidence_path.open("w", encoding="utf-8") as f:
        f.write("# Benchmark Evidence\n\n")
        f.write(f"- Profile: **{profile}**\n")
        f.write("- Fairness policy: same train/val split, same target, fixed random seed (`42`).\n")
        f.write("\n## Tool Status\n")
        for row in benchmark_status_rows:
            f.write(
                f"- {row['tool']}: {row['status']} ({row['rows']} rows)"
                + (f" — {row['notes']}" if row["notes"] else "")
                + "\n"
            )
        f.write("\n## Interpretation\n")
        f.write(
            "- Manual model ranking is always generated and drives model selection.\n"
            "- AutoML tools complement analysis, not replace split discipline or metric interpretation.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark California Housing regressors.")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="quick = deterministic fast baseline; full = includes AutoML benchmark tools.",
    )
    parser.add_argument(
        "--flaml-seconds",
        type=int,
        default=90,
        help="Time budget (seconds) for FLAML AutoML benchmark when profile=full.",
    )
    parser.add_argument(
        "--automl-max-train-rows",
        type=int,
        default=4000,
        help="Maximum sampled train rows used by AutoML benchmark tools in full profile.",
    )
    args = parser.parse_args()

    full_profile = args.profile == "full"

    print("[1/8] Loading dataset...")
    X, y = load_california_housing()
    split = split_dataset(X, y, random_state=GLOBAL_SEED)

    print("[2/8] Training candidate models and selecting best...")
    trained, ranking = train_best_model(
        X_train=split.X_train,
        y_train=split.y_train,
        X_val=split.X_val,
        y_val=split.y_val,
        X_test=split.X_test,
        y_test=split.y_test,
        random_state=GLOBAL_SEED,
    )

    print("[3/8] Saving model + metadata...")
    metadata = build_metadata(
        trained=trained,
        ranking=ranking,
        n_train=len(split.X_train),
        n_val=len(split.X_val),
        n_test=len(split.X_test),
        feature_schema_version="california-housing-v1",
    )
    metadata["trained_at_utc"] = datetime.now(UTC).isoformat()
    metadata["serialization_format"] = "joblib"
    metadata["benchmark_profile"] = args.profile
    metadata["security_note"] = (
        "Never load untrusted pickle/joblib artifacts in production. "
        "Use signed artifacts and trusted storage only."
    )

    save_model(trained.model, MODEL_PATH)
    onnx_ok, onnx_message = export_onnx_if_available(
        trained.model,
        ARTIFACTS_DIR / "reports" / "model.onnx",
        n_features=split.X_train.shape[1],
    )
    metadata["onnx_exported"] = onnx_ok
    metadata["onnx_note"] = onnx_message
    save_metadata(metadata, METADATA_PATH)

    print("[4/8] Writing training reports...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(BENCHMARKS_DIR / "model_ranking.csv", index=False)
    dump_training_summary(metadata, REPORTS_DIR / "training_summary.json")

    print("[5/8] Running AutoML/benchmark tools (LazyPredict, FLAML, PyCaret)...")
    benchmark_status_rows = []
    benchmark_status_rows.append(
        _save_benchmark_result(
            run_lazypredict_benchmark(
                X_train=split.X_train,
                X_val=split.X_val,
                y_train=split.y_train,
                y_val=split.y_val,
                enabled=full_profile,
                max_train_rows=args.automl_max_train_rows,
            )
        )
    )
    benchmark_status_rows.append(
        _save_benchmark_result(
            run_flaml_benchmark(
                X_train=split.X_train,
                X_val=split.X_val,
                y_train=split.y_train,
                y_val=split.y_val,
                time_budget_seconds=args.flaml_seconds,
                enabled=full_profile,
                max_train_rows=args.automl_max_train_rows,
            )
        )
    )
    benchmark_status_rows.append(
        _save_benchmark_result(
            run_pycaret_benchmark(
                X_train=split.X_train,
                X_val=split.X_val,
                y_train=split.y_train,
                y_val=split.y_val,
                enabled=full_profile,
                max_train_rows=args.automl_max_train_rows,
            )
        )
    )

    pd.DataFrame(benchmark_status_rows).to_csv(BENCHMARKS_DIR / "benchmark_status.csv", index=False)
    _write_benchmark_evidence(args.profile, benchmark_status_rows)

    print("[6/8] Generating figures...")
    plot_model_ranking(ranking, FIGURES_DIR / "model_comparison_rmse.png")
    plot_metric_radar(ranking, FIGURES_DIR / "best_model_metrics.png")

    print("[7/8] Writing quick summary markdown...")
    summary_path = REPORTS_DIR / "training_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Training Summary\n\n")
        f.write(f"- Best model: **{trained.model_name}**\n")
        f.write(f"- Test RMSE: **{trained.test_metrics.rmse:.4f}**\n")
        f.write(f"- Test R2: **{trained.test_metrics.r2:.4f}**\n")
        f.write(f"- Benchmark profile: **{args.profile}**\n")
        f.write(f"- Artifact: `{MODEL_PATH}`\n")
        f.write(f"- Metadata: `{METADATA_PATH}`\n")
        f.write("\n## Benchmark tool status\n")
        for row in benchmark_status_rows:
            f.write(
                f"- {row['tool']}: {row['status']} ({row['rows']} rows)"
                + (f" — {row['notes']}" if row["notes"] else "")
                + "\n"
            )

    print("[8/8] Final summary")
    print("Done. Artifacts generated under artifacts/ and models/.")
    print(
        json.dumps(
            {
                "best_model": trained.model_name,
                "test_rmse": trained.test_metrics.rmse,
                "profile": args.profile,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
