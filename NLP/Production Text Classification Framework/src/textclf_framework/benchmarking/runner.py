"""Model benchmark matrix runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(slots=True)
class MatrixRow:
    timestamp_utc: str
    dataset: str
    model: str
    strategy: str
    accuracy: float
    macro_f1: float
    train_seconds: float
    inference_latency_ms: float
    gpu_memory_mb: float
    cpu_latency_ms: float
    model_size_mb: float


class BenchmarkMatrixRunner:
    """Run and persist benchmark matrix rows across dataset/model combinations."""

    def __init__(self) -> None:
        self.rows: list[MatrixRow] = []

    def add_row(self, row: MatrixRow) -> None:
        self.rows.append(row)

    def run(
        self,
        datasets: list[str],
        models: list[str],
        evaluator: Callable[[str, str], dict[str, float]],
        strategy_resolver: Callable[[str], str],
    ) -> pd.DataFrame:
        """Run benchmark evaluator callback for every dataset/model pair."""
        for dataset in datasets:
            for model in models:
                metrics = evaluator(dataset, model)
                self.add_row(
                    MatrixRow(
                        timestamp_utc=datetime.utcnow().isoformat(),
                        dataset=dataset,
                        model=model,
                        strategy=strategy_resolver(model),
                        accuracy=float(metrics.get("accuracy", 0.0)),
                        macro_f1=float(metrics.get("macro_f1", 0.0)),
                        train_seconds=float(metrics.get("train_seconds", 0.0)),
                        inference_latency_ms=float(metrics.get("inference_latency_ms", 0.0)),
                        gpu_memory_mb=float(metrics.get("gpu_memory_mb", 0.0)),
                        cpu_latency_ms=float(metrics.get("cpu_latency_ms", 0.0)),
                        model_size_mb=float(metrics.get("model_size_mb", 0.0)),
                    )
                )
        return pd.DataFrame([asdict(r) for r in self.rows])

    def save(self, output_path: str | Path) -> Path:
        """Save benchmark matrix CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(r) for r in self.rows]).to_csv(path, index=False)
        return path
