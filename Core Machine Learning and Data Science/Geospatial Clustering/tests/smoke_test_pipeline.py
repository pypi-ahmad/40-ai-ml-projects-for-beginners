"""Smoke tests for geospatial clustering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering import run_algorithm
from src.config import ASSIGNMENTS_PATH, CLUSTERING_DEFAULTS, TRAIN_FILE_PATH
from src.pipeline import GeospatialClusteringPipeline


def test_defaults_exist() -> None:
    assert "dbscan" in CLUSTERING_DEFAULTS
    assert CLUSTERING_DEFAULTS["dbscan"]["eps_km"] == "auto"


def test_algorithms_run_on_synthetic() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 6))

    for algo in ["kmeans", "minibatch_kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"]:
        result = run_algorithm(algo, X)
        assert len(result.labels) == len(X)
        assert result.n_clusters >= 0


def test_pipeline_smoke_run(tmp_path) -> None:
    sample_path = tmp_path / "train_sample.csv"
    pd.read_csv(TRAIN_FILE_PATH, nrows=2000).to_csv(sample_path, index=False)

    pipe = GeospatialClusteringPipeline(
        data_path=sample_path,
        remove_outliers=True,
        algorithms=["kmeans", "minibatch_kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"],
        run_downstream_automl=False,
    )
    report = pipe.run()

    assert report.n_raw_rows > 0
    assert report.n_clean_rows > 0
    assert report.n_samples > 0
    assert report.n_features > 0
    assert len(report.algorithm_results) >= 3
    assert report.best_algorithm is not None
    assert ASSIGNMENTS_PATH.exists()
