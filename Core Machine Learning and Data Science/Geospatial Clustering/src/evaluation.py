"""Evaluation utilities for clustering quality, stability, and business impact."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.clustering import ClusteringResult, kmeans_clustering, run_algorithm
from src.config import (
    COL_CLUSTER,
    COL_DELIVERY_DISTANCE,
    EVALUATION_METRICS,
    K_SELECTION_RANGE,
)

logger = logging.getLogger(__name__)


def silhouette_score(x_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score with safe handling for noise/single-cluster cases."""
    from sklearn.metrics import silhouette_score as _silhouette_score

    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")
    return float(_silhouette_score(x_matrix[mask], labels[mask]))


def davies_bouldin_index(x_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute Davies-Bouldin index (lower is better)."""
    from sklearn.metrics import davies_bouldin_score

    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")
    return float(davies_bouldin_score(x_matrix[mask], labels[mask]))


def calinski_harabasz_index(x_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute Calinski-Harabasz index (higher is better)."""
    from sklearn.metrics import calinski_harabasz_score

    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return float("nan")
    return float(calinski_harabasz_score(x_matrix[mask], labels[mask]))


def _clean_result_params(params: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in params.items()
        if not key.startswith("_") and key != "geo_coords"
    }


def _pairwise_ari(labels_list: list[np.ndarray]) -> float:
    from sklearn.metrics import adjusted_rand_score

    if len(labels_list) < 2:
        return 1.0

    scores: list[float] = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            scores.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
    return float(np.mean(scores)) if scores else 1.0


def estimate_stability(
    algorithm: str,
    x_matrix: np.ndarray,
    *,
    geo_coords: np.ndarray | None,
    params: dict[str, object],
    random_state: int = 42,
) -> float:
    """Estimate clustering stability with repeated fits and ARI comparison."""
    deterministic = {"dbscan", "hdbscan", "agglomerative"}
    if algorithm in deterministic:
        return 1.0

    rng = np.random.default_rng(random_state)
    subset_n = min(len(x_matrix), 6000)
    subset_idx = np.sort(rng.choice(len(x_matrix), size=subset_n, replace=False))

    x_sub = x_matrix[subset_idx]
    geo_sub = geo_coords[subset_idx] if geo_coords is not None else None

    labels_list: list[np.ndarray] = []
    for seed in [random_state, random_state + 1, random_state + 2]:
        run_params = dict(params)
        run_params["random_state"] = seed
        result = run_algorithm(algorithm, x_sub, geo_coords=geo_sub, **run_params)
        labels_list.append(result.labels)

    return _pairwise_ari(labels_list)


def _minmax_norm(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return pd.Series(np.zeros(len(values)), index=values.index)

    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value - min_value < 1e-12:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def evaluate_all(
    x_matrix: np.ndarray,
    results: dict[str, ClusteringResult],
    *,
    geo_coords: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate multiple clustering outputs with standardized metrics and stability."""
    rows: list[dict[str, float | str]] = []

    for algorithm, result in results.items():
        sil = silhouette_score(x_matrix, result.labels)
        db_index = davies_bouldin_index(x_matrix, result.labels)
        ch_index = calinski_harabasz_index(x_matrix, result.labels)
        stability = estimate_stability(
            algorithm,
            x_matrix,
            geo_coords=geo_coords,
            params=_clean_result_params(result.params),
        )
        noise_ratio = float(result.n_noise / max(len(result.labels), 1))

        result.score = sil
        rows.append(
            {
                "algorithm": algorithm,
                "n_clusters": float(result.n_clusters),
                "n_noise": float(result.n_noise),
                "noise_ratio": noise_ratio,
                "silhouette": sil,
                "davies_bouldin": db_index,
                "calinski_harabasz": ch_index,
                "stability_ari": stability,
            }
        )

        logger.info(
            "%s -> silhouette=%.4f, db=%.4f, ch=%.2f, stability=%.3f, clusters=%d",
            algorithm,
            sil,
            db_index,
            ch_index,
            stability,
            result.n_clusters,
        )

    eval_df = pd.DataFrame(rows)
    if len(eval_df) == 0:
        return eval_df

    sil_n = _minmax_norm(eval_df["silhouette"].fillna(eval_df["silhouette"].min()))
    db_n = _minmax_norm(eval_df["davies_bouldin"].fillna(eval_df["davies_bouldin"].max()))
    ch_n = _minmax_norm(eval_df["calinski_harabasz"].fillna(eval_df["calinski_harabasz"].min()))
    st_n = _minmax_norm(eval_df["stability_ari"].fillna(0.0))
    noise_n = _minmax_norm(eval_df["noise_ratio"].fillna(1.0))

    eval_df["composite_score"] = sil_n + (1 - db_n) + ch_n + st_n + (1 - noise_n)
    return eval_df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)


def k_selection_diagnostics(
    x_matrix: np.ndarray,
    *,
    k_values: range = K_SELECTION_RANGE,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate Elbow + Silhouette + DB + CH diagnostics across candidate K."""
    rows: list[dict[str, float]] = []

    for k_value in k_values:
        result = kmeans_clustering(x_matrix, n_clusters=int(k_value), random_state=random_state)
        labels = result.labels
        rows.append(
            {
                "k": float(k_value),
                "inertia": float(result.model.inertia_),
                "silhouette": silhouette_score(x_matrix, labels),
                "davies_bouldin": davies_bouldin_index(x_matrix, labels),
                "calinski_harabasz": calinski_harabasz_index(x_matrix, labels),
            }
        )

    return pd.DataFrame(rows)


def elbow_curve(
    x_matrix: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Backward-compatible elbow helper used by existing plotting functions."""
    diagnostics = k_selection_diagnostics(x_matrix, k_values=k_range, random_state=random_state)
    return diagnostics.rename(columns={"k": "n_clusters"})[["n_clusters", "inertia", "silhouette"]]


def compute_business_impact_metrics(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    outlier_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute business impact summary from clustered delivery data."""
    temp = df.copy()
    temp[COL_CLUSTER] = labels

    cluster_counts = temp[COL_CLUSTER].value_counts(dropna=False)
    avg_distance = float(temp[COL_DELIVERY_DISTANCE].mean()) if COL_DELIVERY_DISTANCE in temp else float("nan")
    p90_distance = (
        float(temp[COL_DELIVERY_DISTANCE].quantile(0.9))
        if COL_DELIVERY_DISTANCE in temp
        else float("nan")
    )

    if outlier_mask is None:
        outlier_pct = 0.0
    else:
        outlier_pct = float(np.mean(outlier_mask) * 100)

    zone_coverage_pct = float((cluster_counts / max(len(temp), 1)).sum() * 100)

    return {
        "avg_delivery_distance_km": avg_distance,
        "p90_delivery_distance_km": p90_distance,
        "cluster_count": float(len([cluster for cluster in cluster_counts.index if cluster != -1])),
        "noise_point_pct": float((cluster_counts.get(-1, 0) / max(len(temp), 1)) * 100),
        "outlier_pct": outlier_pct,
        "coverage_pct": zone_coverage_pct,
    }


def evaluate_metric_table(df: pd.DataFrame) -> dict[str, float]:
    """Return compact scalar summary for report cards."""
    if df.empty:
        metrics = {metric: float("nan") for metric in EVALUATION_METRICS}
        metrics["stability_ari"] = float("nan")
        metrics["composite_score"] = float("nan")
        return metrics

    best = df.iloc[0]
    return {
        "silhouette": float(best.get("silhouette", float("nan"))),
        "davies_bouldin": float(best.get("davies_bouldin", float("nan"))),
        "calinski_harabasz": float(best.get("calinski_harabasz", float("nan"))),
        "stability_ari": float(best.get("stability_ari", float("nan"))),
        "composite_score": float(best.get("composite_score", float("nan"))),
    }
