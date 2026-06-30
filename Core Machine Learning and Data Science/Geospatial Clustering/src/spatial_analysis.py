"""Advanced spatial analytics: heatmaps, KDE hotspots, and coverage metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    COL_CLUSTER,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
)
from src.distance import haversine_vectorized


@dataclass
class CoverageSummary:
    """Coverage statistics for one cluster."""

    cluster: int
    count: int
    centroid_lat: float
    centroid_lon: float
    median_distance_to_centroid_km: float
    p90_distance_to_centroid_km: float
    within_service_radius_pct: float


def build_grid_density(
    df: pd.DataFrame,
    *,
    lat_col: str = COL_RESTAURANT_LAT,
    lon_col: str = COL_RESTAURANT_LON,
    bins: int = 80,
) -> pd.DataFrame:
    """Build a grid-density table suitable for heatmaps."""
    lat_values = df[lat_col].to_numpy(dtype=float)
    lon_values = df[lon_col].to_numpy(dtype=float)

    hist, lat_edges, lon_edges = np.histogram2d(lat_values, lon_values, bins=bins)

    rows: list[dict[str, float]] = []
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            count = hist[i, j]
            if count <= 0:
                continue
            rows.append(
                {
                    "lat_min": float(lat_edges[i]),
                    "lat_max": float(lat_edges[i + 1]),
                    "lon_min": float(lon_edges[j]),
                    "lon_max": float(lon_edges[j + 1]),
                    "count": float(count),
                }
            )

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def kde_hotspots(
    df: pd.DataFrame,
    *,
    lat_col: str = COL_RESTAURANT_LAT,
    lon_col: str = COL_RESTAURANT_LON,
    bandwidth: float = 0.08,
    top_n: int = 150,
) -> pd.DataFrame:
    """Estimate demand hotspots using Kernel Density Estimation."""
    from sklearn.neighbors import KernelDensity

    coords = df[[lat_col, lon_col]].to_numpy(dtype=float)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(coords)

    density_scores = kde.score_samples(coords)
    out = df[[lat_col, lon_col]].copy()
    out["kde_log_density"] = density_scores
    out["kde_density_rank"] = out["kde_log_density"].rank(ascending=False, method="dense")

    return out.sort_values("kde_log_density", ascending=False).head(top_n).reset_index(drop=True)


def service_coverage_analysis(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    service_radius_km: float = 5.0,
    lat_col: str = COL_DELIVERY_LAT,
    lon_col: str = COL_DELIVERY_LON,
) -> pd.DataFrame:
    """Measure how tightly each cluster is covered around its centroid."""
    temp = df.copy()
    temp[COL_CLUSTER] = labels

    summaries: list[CoverageSummary] = []
    for cluster in sorted(set(labels)):
        if cluster == -1:
            continue
        subset = temp[temp[COL_CLUSTER] == cluster]
        if subset.empty:
            continue

        centroid_lat = float(subset[lat_col].mean())
        centroid_lon = float(subset[lon_col].mean())

        dists = haversine_vectorized(
            subset[lat_col].to_numpy(dtype=float),
            subset[lon_col].to_numpy(dtype=float),
            np.full(len(subset), centroid_lat),
            np.full(len(subset), centroid_lon),
        )

        summaries.append(
            CoverageSummary(
                cluster=int(cluster),
                count=int(len(subset)),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                median_distance_to_centroid_km=float(np.median(dists)),
                p90_distance_to_centroid_km=float(np.percentile(dists, 90)),
                within_service_radius_pct=float(np.mean(dists <= service_radius_km) * 100),
            )
        )

    return pd.DataFrame([s.__dict__ for s in summaries])
