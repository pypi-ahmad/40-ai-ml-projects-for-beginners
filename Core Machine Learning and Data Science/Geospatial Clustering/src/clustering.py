"""Clustering algorithms and geospatial cluster geometry helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import CLUSTERING_DEFAULTS
from src.distance import EARTH_RADIUS_KM

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Standard result contract for all clustering algorithms."""

    name: str
    labels: np.ndarray
    model: Any
    n_clusters: int
    n_noise: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


def _scale_features(x_matrix: np.ndarray) -> tuple[np.ndarray, Any]:
    """Scale features before distance-sensitive clustering methods."""
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    return scaler.fit_transform(x_matrix), scaler


def _estimate_eps(x_matrix: np.ndarray, *, min_samples: int = 20) -> float:
    """Estimate DBSCAN epsilon from sorted k-nearest-neighbor distances."""
    from sklearn.neighbors import NearestNeighbors

    k_neighbors = min(max(2, min_samples), max(2, len(x_matrix) - 1))
    nn_model = NearestNeighbors(n_neighbors=k_neighbors)
    nn_model.fit(x_matrix)
    distances, _ = nn_model.kneighbors(x_matrix)
    k_dist = np.sort(distances[:, -1])

    x = np.linspace(0.0, 1.0, len(k_dist))
    y = (k_dist - k_dist.min()) / (k_dist.max() - k_dist.min() + 1e-12)

    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return float(np.percentile(k_dist, 90))

    line = line / norm
    start = np.array([x[0], y[0]])
    distances_to_line = []
    for xi, yi in zip(x, y, strict=False):
        point = np.array([xi, yi])
        proj = start + line * np.dot(point - start, line)
        distances_to_line.append(np.linalg.norm(point - proj))

    idx = int(np.argmax(distances_to_line))
    eps = float(k_dist[idx])
    logger.info("Estimated DBSCAN eps=%.4f (scaled-feature space)", eps)
    return eps


def _estimate_eps_geo_km(geo_coords_rad: np.ndarray, *, min_samples: int = 20) -> float:
    """Estimate DBSCAN eps in kilometers on haversine metric space."""
    from sklearn.neighbors import NearestNeighbors

    k_neighbors = min(max(2, min_samples), max(2, len(geo_coords_rad) - 1))
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="haversine")
    nn_model.fit(geo_coords_rad)
    distances_rad, _ = nn_model.kneighbors(geo_coords_rad)
    k_dist_km = np.sort(distances_rad[:, -1] * EARTH_RADIUS_KM)

    x = np.linspace(0.0, 1.0, len(k_dist_km))
    y = (k_dist_km - k_dist_km.min()) / (k_dist_km.max() - k_dist_km.min() + 1e-12)

    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return float(np.percentile(k_dist_km, 90))

    line = line / norm
    start = np.array([x[0], y[0]])
    distances_to_line = []
    for xi, yi in zip(x, y, strict=False):
        point = np.array([xi, yi])
        proj = start + line * np.dot(point - start, line)
        distances_to_line.append(np.linalg.norm(point - proj))

    idx = int(np.argmax(distances_to_line))
    eps_km = float(np.clip(k_dist_km[idx], 0.5, 30.0))
    logger.info("Estimated DBSCAN eps=%.4f km (haversine geo space)", eps_km)
    return eps_km


def kmeans_clustering(
    x_matrix: np.ndarray,
    *,
    n_clusters: int = 6,
    random_state: int = 42,
    n_init: str | int = "auto",
) -> ClusteringResult:
    """Run K-Means clustering."""
    from sklearn.cluster import KMeans

    x_scaled, scaler = _scale_features(x_matrix)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(x_scaled)

    return ClusteringResult(
        name="K-Means",
        labels=labels,
        model=model,
        n_clusters=int(model.n_clusters),
        params={
            "n_clusters": n_clusters,
            "random_state": random_state,
            "n_init": n_init,
            "_scaler": scaler,
        },
    )


def minibatch_kmeans_clustering(
    x_matrix: np.ndarray,
    *,
    n_clusters: int = 6,
    batch_size: int = 2048,
    random_state: int = 42,
    n_init: str | int = "auto",
) -> ClusteringResult:
    """Run MiniBatch K-Means clustering."""
    from sklearn.cluster import MiniBatchKMeans

    x_scaled, scaler = _scale_features(x_matrix)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(x_scaled)

    return ClusteringResult(
        name="MiniBatch-KMeans",
        labels=labels,
        model=model,
        n_clusters=int(model.n_clusters),
        params={
            "n_clusters": n_clusters,
            "batch_size": batch_size,
            "random_state": random_state,
            "n_init": n_init,
            "_scaler": scaler,
        },
    )


def dbscan_clustering(
    x_matrix: np.ndarray,
    *,
    geo_coords: np.ndarray | None = None,
    eps_km: float | str = "auto",
    min_samples: int = 30,
) -> ClusteringResult:
    """Run DBSCAN clustering with geospatial haversine support."""
    from sklearn.cluster import DBSCAN

    if geo_coords is not None:
        geo_coords_rad = np.radians(np.asarray(geo_coords, dtype=float))
        effective_eps_km = (
            _estimate_eps_geo_km(geo_coords_rad, min_samples=min_samples)
            if eps_km == "auto"
            else float(eps_km)
        )
        model = DBSCAN(
            eps=effective_eps_km / EARTH_RADIUS_KM,
            min_samples=min_samples,
            metric="haversine",
        )
        labels = model.fit_predict(geo_coords_rad)
        params = {
            "eps_km": effective_eps_km,
            "min_samples": min_samples,
            "metric": "haversine",
        }
    else:
        x_scaled, scaler = _scale_features(x_matrix)
        effective_eps = _estimate_eps(x_scaled, min_samples=min_samples) if eps_km == "auto" else float(eps_km)
        model = DBSCAN(eps=effective_eps, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)
        params = {
            "eps_scaled": effective_eps,
            "min_samples": min_samples,
            "metric": "euclidean",
            "_scaler": scaler,
        }

    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int(np.sum(labels == -1))

    return ClusteringResult(
        name="DBSCAN",
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        n_noise=n_noise,
        params=params,
    )


def hdbscan_clustering(
    x_matrix: np.ndarray,
    *,
    geo_coords: np.ndarray | None = None,
    min_cluster_size: int = 120,
    min_samples: int = 40,
) -> ClusteringResult:
    """Run HDBSCAN clustering. Falls back to DBSCAN when unavailable."""
    try:
        import hdbscan
    except Exception:
        logger.warning("hdbscan import failed; falling back to DBSCAN")
        return dbscan_clustering(x_matrix, geo_coords=geo_coords, eps_km="auto", min_samples=min_samples)

    if geo_coords is not None:
        geo_coords_rad = np.radians(np.asarray(geo_coords, dtype=float))
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="haversine",
        )
        labels = model.fit_predict(geo_coords_rad)
        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "haversine",
        }
    else:
        x_scaled, scaler = _scale_features(x_matrix)
        model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)
        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "euclidean",
            "_scaler": scaler,
        }

    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int(np.sum(labels == -1))

    return ClusteringResult(
        name="HDBSCAN",
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        n_noise=n_noise,
        params=params,
    )


def agglomerative_clustering(
    x_matrix: np.ndarray,
    *,
    n_clusters: int = 6,
    linkage: str = "ward",
) -> ClusteringResult:
    """Run agglomerative hierarchical clustering."""
    from sklearn.cluster import AgglomerativeClustering

    x_scaled, scaler = _scale_features(x_matrix)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(x_scaled)

    return ClusteringResult(
        name="Agglomerative",
        labels=labels,
        model=model,
        n_clusters=n_clusters,
        params={"n_clusters": n_clusters, "linkage": linkage, "_scaler": scaler},
    )


def gmm_clustering(
    x_matrix: np.ndarray,
    *,
    n_components: int = 6,
    covariance_type: str = "full",
    random_state: int = 42,
) -> ClusteringResult:
    """Run Gaussian Mixture Model clustering."""
    from sklearn.mixture import GaussianMixture

    x_scaled, scaler = _scale_features(x_matrix)
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = model.fit_predict(x_scaled)

    return ClusteringResult(
        name="GMM",
        labels=labels,
        model=model,
        n_clusters=n_components,
        params={
            "n_components": n_components,
            "covariance_type": covariance_type,
            "random_state": random_state,
            "_scaler": scaler,
        },
    )


ALGORITHM_MAP: dict[str, Any] = {
    "kmeans": kmeans_clustering,
    "minibatch_kmeans": minibatch_kmeans_clustering,
    "dbscan": dbscan_clustering,
    "hdbscan": hdbscan_clustering,
    "agglomerative": agglomerative_clustering,
    "gmm": gmm_clustering,
}


def run_algorithm(
    name: str,
    x_matrix: np.ndarray,
    *,
    geo_coords: np.ndarray | None = None,
    **kwargs: Any,
) -> ClusteringResult:
    """Run one clustering algorithm by canonical name."""
    if name not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHM_MAP)}")

    params = {**CLUSTERING_DEFAULTS.get(name, {}), **kwargs}
    if name in {"dbscan", "hdbscan"}:
        params["geo_coords"] = geo_coords

    logger.info("Running %s with params=%s", name, {k: v for k, v in params.items() if k != "geo_coords"})
    return ALGORITHM_MAP[name](x_matrix, **params)


def run_all(
    x_matrix: np.ndarray,
    *,
    geo_coords: np.ndarray | None = None,
    algorithms: list[str] | None = None,
    shared_kwargs: dict[str, Any] | None = None,
) -> dict[str, ClusteringResult]:
    """Run all or selected algorithms and collect results."""
    algos = algorithms or list(ALGORITHM_MAP.keys())
    shared = shared_kwargs or {}
    results: dict[str, ClusteringResult] = {}

    for algo in algos:
        results[algo] = run_algorithm(algo, x_matrix, geo_coords=geo_coords, **shared)
    return results


def cluster_centroids(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    labels: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Compute centroid latitude/longitude per non-noise cluster."""
    centroids: dict[int, tuple[float, float]] = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        mask = labels == label
        centroids[int(label)] = (
            float(np.mean(latitudes[mask])),
            float(np.mean(longitudes[mask])),
        )
    return centroids


def cluster_boundaries(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    labels: np.ndarray,
) -> dict[int, list[tuple[float, float]]]:
    """Approximate cluster boundaries with convex hull coordinates."""
    from scipy.spatial import ConvexHull

    boundaries: dict[int, list[tuple[float, float]]] = {}
    points = np.column_stack([latitudes, longitudes])

    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_points = points[labels == label]
        if len(cluster_points) > 5000:
            rng = np.random.default_rng(42 + int(label))
            idx = rng.choice(len(cluster_points), size=5000, replace=False)
            cluster_points = cluster_points[np.sort(idx)]

        if len(cluster_points) < 3:
            boundaries[int(label)] = [
                (float(lat), float(lon))
                for lat, lon in cluster_points
            ]
            continue

        try:
            hull = ConvexHull(cluster_points)
            poly = [tuple(map(float, cluster_points[idx])) for idx in hull.vertices]
            boundaries[int(label)] = poly
        except Exception:
            boundaries[int(label)] = [
                (float(lat), float(lon))
                for lat, lon in cluster_points[:10]
            ]

    return boundaries
