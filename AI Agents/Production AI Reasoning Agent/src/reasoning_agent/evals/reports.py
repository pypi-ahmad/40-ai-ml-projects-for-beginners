"""Benchmark report serialization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path

import orjson
import polars as pl

from reasoning_agent.evals.benchmark import BenchmarkSummary
from reasoning_agent.evals.dataset import BenchmarkPrediction


def save_benchmark_reports(
    output_dir: Path,
    predictions: list[BenchmarkPrediction],
    summaries: list[BenchmarkSummary],
) -> dict[str, Path]:
    """Save benchmark outputs in JSON/CSV/Parquet."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    pred_path = output_dir / f"predictions_{timestamp}.jsonl"
    with pred_path.open("wb") as handle:
        for prediction in predictions:
            handle.write(orjson.dumps(prediction.model_dump(mode="json")) + b"\n")

    summary_json = output_dir / f"summary_{timestamp}.json"
    summary_payload = [asdict(summary) for summary in summaries]
    summary_json.write_bytes(orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2))

    summary_df = pl.DataFrame(summary_payload)
    summary_csv = output_dir / f"summary_{timestamp}.csv"
    summary_parquet = output_dir / f"summary_{timestamp}.parquet"
    summary_df.write_csv(summary_csv)
    summary_df.write_parquet(summary_parquet)

    return {
        "predictions_jsonl": pred_path,
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "summary_parquet": summary_parquet,
    }
