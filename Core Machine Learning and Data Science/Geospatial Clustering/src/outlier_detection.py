"""Outlier detection methods for geospatial clustering workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_DELIVERY_DISTANCE,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_DURATION_MIN,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_SPEED_KMPH,
    COL_TIME_TAKEN,
    DEFAULT_OUTLIER_CONTAMINATION,
    INDIA_LAT_MAX,
    INDIA_LAT_MIN,
    INDIA_LON_MAX,
    INDIA_LON_MIN,
    OUTLIER_METHODS,
)
from src.distance import haversine_vectorized

logger = logging.getLogger(__name__)


@dataclass
class OutlierReport:
    """Result from one outlier detector."""

    name: str
    mask: np.ndarray
    n_outliers: int
    details: dict[str, Any] = field(default_factory=dict)


def coordinate_bounds_outliers(df: pd.DataFrame) -> OutlierReport:
    """Flag coordinates outside India bounds."""
    mask = (
        (df[COL_RESTAURANT_LAT] < INDIA_LAT_MIN)
        | (df[COL_RESTAURANT_LAT] > INDIA_LAT_MAX)
        | (df[COL_RESTAURANT_LON] < INDIA_LON_MIN)
        | (df[COL_RESTAURANT_LON] > INDIA_LON_MAX)
        | (df[COL_DELIVERY_LAT] < INDIA_LAT_MIN)
        | (df[COL_DELIVERY_LAT] > INDIA_LAT_MAX)
        | (df[COL_DELIVERY_LON] < INDIA_LON_MIN)
        | (df[COL_DELIVERY_LON] > INDIA_LON_MAX)
    ).to_numpy()

    return OutlierReport(
        name="coordinate_bounds",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"bounds": {"lat": [INDIA_LAT_MIN, INDIA_LAT_MAX], "lon": [INDIA_LON_MIN, INDIA_LON_MAX]}},
    )


def gps_error_outliers(df: pd.DataFrame, *, max_distance_km: float = 60.0) -> OutlierReport:
    """Flag likely GPS errors (zero coordinates or impossible delivery distance)."""
    zero_mask = (
        (df[COL_RESTAURANT_LAT] == 0)
        | (df[COL_RESTAURANT_LON] == 0)
        | (df[COL_DELIVERY_LAT] == 0)
        | (df[COL_DELIVERY_LON] == 0)
    ).to_numpy()

    distances = haversine_vectorized(
        df[COL_RESTAURANT_LAT].to_numpy(dtype=float),
        df[COL_RESTAURANT_LON].to_numpy(dtype=float),
        df[COL_DELIVERY_LAT].to_numpy(dtype=float),
        df[COL_DELIVERY_LON].to_numpy(dtype=float),
    )
    extreme_distance_mask = distances > max_distance_km
    mask = zero_mask | extreme_distance_mask

    return OutlierReport(
        name="gps_error",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={
            "zero_coordinate_rows": int(zero_mask.sum()),
            "extreme_distance_rows": int(extreme_distance_mask.sum()),
            "max_distance_threshold_km": max_distance_km,
        },
    )


def iqr_outliers(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    multiplier: float = 1.5,
) -> OutlierReport:
    """Detect univariate outliers by Tukey IQR rule."""
    cols = columns or [COL_DELIVERY_DISTANCE, COL_SPEED_KMPH, COL_DURATION_MIN, COL_TIME_TAKEN]
    cols = [c for c in cols if c in df.columns]

    mask = np.zeros(len(df), dtype=bool)
    thresholds: dict[str, tuple[float, float]] = {}

    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask |= (values < lower) | (values > upper)
        thresholds[col] = (lower, upper)

    return OutlierReport(
        name="iqr",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"columns": cols, "multiplier": multiplier, "thresholds": thresholds},
    )


def mahalanobis_outliers(X: np.ndarray, *, quantile: float = 0.999) -> OutlierReport:
    """Detect multivariate outliers using Mahalanobis distance."""
    from scipy.stats import chi2

    n_samples, n_features = X.shape
    threshold = float(np.sqrt(chi2.ppf(quantile, df=n_features)))

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov = cov + np.eye(n_features) * 1e-6
        inv_cov = np.linalg.inv(cov)

    diff = X - mean
    dist_sq = np.sum(diff @ inv_cov * diff, axis=1)
    dist = np.sqrt(np.clip(dist_sq, 0.0, None))
    mask = dist > threshold

    return OutlierReport(
        name="mahalanobis",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"threshold": threshold, "quantile": quantile, "n_samples": n_samples},
    )


def isolation_forest_outliers(
    X: np.ndarray,
    *,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    random_state: int = 42,
) -> OutlierReport:
    """Detect anomalies using Isolation Forest."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(X)
    mask = preds == -1

    return OutlierReport(
        name="isolation_forest",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"contamination": contamination},
    )


def local_outlier_factor_outliers(
    X: np.ndarray,
    *,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
    n_neighbors: int = 35,
) -> OutlierReport:
    """Detect local-density anomalies with LOF."""
    from sklearn.neighbors import LocalOutlierFactor

    n_neighbors = min(max(5, n_neighbors), max(5, len(X) - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = model.fit_predict(X)
    mask = preds == -1

    return OutlierReport(
        name="lof",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"contamination": contamination, "n_neighbors": n_neighbors},
    )


def _estimate_eps(X: np.ndarray, *, k: int = 20) -> float:
    """Estimate DBSCAN epsilon from k-distance elbow heuristic."""
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    k = min(max(2, k), max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = np.sort(distances[:, -1])

    x = np.linspace(0.0, 1.0, len(k_dist))
    y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min() + 1e-9)

    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return float(np.percentile(k_dist, 90))

    line_unit = line_vec / line_len
    distances_to_line = []
    for xi, yi in zip(x, y, strict=False):
        p = np.array([xi, yi])
        proj = line_start + line_unit * np.dot(p - line_start, line_unit)
        distances_to_line.append(np.linalg.norm(p - proj))

    elbow_idx = int(np.argmax(distances_to_line))
    return float(k_dist[elbow_idx])


def dbscan_noise_outliers(
    X: np.ndarray,
    *,
    eps: float | str = "auto",
    min_samples: int = 25,
) -> OutlierReport:
    """Treat DBSCAN noise points as geospatial outliers."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    effective_eps = _estimate_eps(X_scaled, k=min_samples) if eps == "auto" else float(eps)
    model = DBSCAN(eps=effective_eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    mask = labels == -1

    return OutlierReport(
        name="dbscan_noise",
        mask=mask,
        n_outliers=int(mask.sum()),
        details={"eps": float(effective_eps), "min_samples": min_samples},
    )


def consensus_outliers(reports: list[OutlierReport], *, min_votes: int = 2) -> OutlierReport:
    """Aggregate multiple detector signals with vote threshold."""
    if not reports:
        return OutlierReport(name="consensus", mask=np.array([], dtype=bool), n_outliers=0)

    votes = np.zeros(len(reports[0].mask), dtype=int)
    for report in reports:
        votes += report.mask.astype(int)

    consensus_mask = votes >= min_votes
    return OutlierReport(
        name=f"consensus_>={min_votes}",
        mask=consensus_mask,
        n_outliers=int(consensus_mask.sum()),
        details={"min_votes": min_votes, "n_methods": len(reports), "max_votes": int(votes.max())},
    )


def detect_outliers(
    df: pd.DataFrame,
    *,
    feature_matrix: Optional[np.ndarray] = None,
    methods: Optional[list[str]] = None,
    min_consensus_votes: int = 2,
    contamination: float = DEFAULT_OUTLIER_CONTAMINATION,
) -> tuple[OutlierReport, list[OutlierReport]]:
    """Run configured outlier detectors and return consensus + details."""
    selected_methods = methods or list(OUTLIER_METHODS)
    reports: list[OutlierReport] = []

    if "coordinate_bounds" in selected_methods:
        reports.append(coordinate_bounds_outliers(df))

    if "gps_error" in selected_methods:
        reports.append(gps_error_outliers(df))

    if "iqr" in selected_methods:
        reports.append(iqr_outliers(df))

    if feature_matrix is not None:
        if "mahalanobis" in selected_methods:
            reports.append(mahalanobis_outliers(feature_matrix))

        if "isolation_forest" in selected_methods:
            reports.append(isolation_forest_outliers(feature_matrix, contamination=contamination))

        if "lof" in selected_methods:
            reports.append(local_outlier_factor_outliers(feature_matrix, contamination=contamination))

        if "dbscan_noise" in selected_methods:
            reports.append(dbscan_noise_outliers(feature_matrix))

    consensus = consensus_outliers(reports, min_votes=min_consensus_votes)

    logger.info(
        "Outlier detection completed with %d methods. consensus_outliers=%d",
        len(reports),
        consensus.n_outliers,
    )
    return consensus, reports
