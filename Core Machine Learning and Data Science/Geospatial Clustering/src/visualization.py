"""Visualization utilities for geospatial clustering outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering import cluster_boundaries, cluster_centroids
from src.config import (
    COL_DELIVERY_DISTANCE,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_SPEED_KMPH,
    PLOTS_DIR,
)

logger = logging.getLogger(__name__)

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _ensure_plot_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _save(fig: plt.Figure, filename: str, *, dpi: int = 160) -> str:
    out = _ensure_plot_dir() / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "Cluster Projection",
    x_label: str = "Feature 1",
    y_label: str = "Feature 2",
    filename: str = "clusters_2d.png",
) -> str:
    """Plot 2D cluster projection."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for label in sorted(set(labels)):
        mask = labels == label
        color = "#bbbbbb" if label == -1 else PALETTE[label % len(PALETTE)]
        name = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.6, c=color, label=name)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2, fontsize=8)

    return _save(fig, filename)


def plot_delivery_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    title: str = "Restaurant Locations by Cluster",
    filename: str = "delivery_map.png",
) -> str:
    """Plot geospatial scatter map using restaurant coordinates."""
    coords = np.column_stack([df[COL_RESTAURANT_LON].to_numpy(), df[COL_RESTAURANT_LAT].to_numpy()])
    return plot_clusters_2d(
        coords,
        labels,
        title=title,
        x_label="Longitude",
        y_label="Latitude",
        filename=filename,
    )


def plot_cluster_boundaries(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    title: str = "Cluster Boundaries",
    filename: str = "cluster_boundaries.png",
) -> str:
    """Plot convex-hull style boundaries for each cluster."""
    lat = df[COL_RESTAURANT_LAT].to_numpy(dtype=float)
    lon = df[COL_RESTAURANT_LON].to_numpy(dtype=float)

    bounds = cluster_boundaries(lat, lon, labels)
    centroids = cluster_centroids(lat, lon, labels)

    fig, ax = plt.subplots(figsize=(11, 8))
    for label in sorted(set(labels)):
        mask = labels == label
        color = "#cccccc" if label == -1 else PALETTE[label % len(PALETTE)]
        ax.scatter(lon[mask], lat[mask], s=8, alpha=0.35, c=color)

    for label, poly in bounds.items():
        if len(poly) < 3:
            continue
        poly_lats = [p[0] for p in poly] + [poly[0][0]]
        poly_lons = [p[1] for p in poly] + [poly[0][1]]
        ax.plot(poly_lons, poly_lats, color=PALETTE[label % len(PALETTE)], linewidth=2)

    for label, (cent_lat, cent_lon) in centroids.items():
        ax.scatter([cent_lon], [cent_lat], s=130, marker="X", c=PALETTE[label % len(PALETTE)], edgecolor="black")
        ax.text(cent_lon, cent_lat, f" C{label}", fontsize=9, fontweight="bold")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.2)

    return _save(fig, filename)


def plot_elbow_curve(
    elbow_df: pd.DataFrame,
    *,
    filename: str = "elbow_curve.png",
) -> str:
    """Plot inertia and silhouette across candidate cluster counts."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    k_col = "n_clusters" if "n_clusters" in elbow_df.columns else "k"

    ax1.plot(elbow_df[k_col], elbow_df["inertia"], marker="o", color="#1f77b4")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(elbow_df[k_col], elbow_df["silhouette"], marker="s", linestyle="--", color="#d62728")
    ax2.set_ylabel("Silhouette", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Elbow and Silhouette Diagnostics")
    ax1.grid(alpha=0.2)
    return _save(fig, filename)


def plot_k_selection_metrics(
    diagnostics_df: pd.DataFrame,
    *,
    filename: str = "k_selection_metrics.png",
) -> str:
    """Plot Silhouette, Davies-Bouldin, and Calinski-Harabasz across K."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(diagnostics_df["k"], diagnostics_df["silhouette"], marker="o", color="#1f77b4")
    axes[0].set_title("Silhouette (higher better)")
    axes[0].set_xlabel("k")

    axes[1].plot(diagnostics_df["k"], diagnostics_df["davies_bouldin"], marker="o", color="#d62728")
    axes[1].set_title("Davies-Bouldin (lower better)")
    axes[1].set_xlabel("k")

    axes[2].plot(diagnostics_df["k"], diagnostics_df["calinski_harabasz"], marker="o", color="#2ca02c")
    axes[2].set_title("Calinski-Harabasz (higher better)")
    axes[2].set_xlabel("k")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle("Choosing Number of Clusters")
    return _save(fig, filename)


def plot_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    filename: str = "silhouette_plot.png",
) -> str:
    """Plot silhouette profile by cluster."""
    from sklearn.metrics import silhouette_samples, silhouette_score

    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(X_valid) < 2 or len(set(labels_valid)) < 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Silhouette plot unavailable (need >=2 clusters)", ha="center", va="center")
        ax.axis("off")
        return _save(fig, filename)

    sil_vals = silhouette_samples(X_valid, labels_valid)
    mean_sil = silhouette_score(X_valid, labels_valid)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for label in sorted(set(labels_valid)):
        vals = np.sort(sil_vals[labels_valid == label])
        y_upper = y_lower + len(vals)
        color = PALETTE[label % len(PALETTE)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=color, alpha=0.7)
        ax.text(-0.05, y_lower + len(vals) / 2, str(label), fontsize=8)
        y_lower = y_upper + 10

    ax.axvline(mean_sil, color="red", linestyle="--", label=f"Mean={mean_sil:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Profile")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    return _save(fig, filename)


def plot_feature_distributions(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    features: Optional[list[str]] = None,
    filename: str = "feature_distributions.png",
) -> str:
    """Plot per-cluster feature distributions."""
    feature_list = features or [COL_DELIVERY_DISTANCE, COL_SPEED_KMPH]

    n = len(feature_list)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, feature in zip(axes, feature_list, strict=False):
        plotted = False
        for label in sorted(set(labels)):
            subset = df.loc[labels == label, feature].dropna() if feature in df.columns else pd.Series(dtype=float)
            if subset.empty:
                continue

            # Cap points per cluster for speed on full-size runs.
            if len(subset) > 6000:
                idx = rng.choice(len(subset), size=6000, replace=False)
                subset = subset.iloc[np.sort(idx)]

            color = "#aaaaaa" if label == -1 else PALETTE[label % len(PALETTE)]
            ax.hist(
                subset.to_numpy(),
                bins=50,
                density=True,
                alpha=0.35,
                label=f"C{label}",
                color=color,
            )
            plotted = True

        ax.set_title(f"{feature} by Cluster")
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend(fontsize=8)

    return _save(fig, filename)


def plot_outlier_map(
    df: pd.DataFrame,
    outlier_mask: np.ndarray,
    *,
    filename: str = "outlier_map.png",
) -> str:
    """Plot inliers vs outliers on coordinate map."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        df.loc[~outlier_mask, COL_RESTAURANT_LON],
        df.loc[~outlier_mask, COL_RESTAURANT_LAT],
        s=8,
        c="#1f77b4",
        alpha=0.35,
        label="Inlier",
    )
    ax.scatter(
        df.loc[outlier_mask, COL_RESTAURANT_LON],
        df.loc[outlier_mask, COL_RESTAURANT_LAT],
        s=10,
        c="#d62728",
        alpha=0.8,
        label="Outlier",
    )
    ax.set_title("Geospatial Outliers")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(alpha=0.2)

    return _save(fig, filename)


def plot_density_heatmap(
    df: pd.DataFrame,
    *,
    lat_col: str = COL_RESTAURANT_LAT,
    lon_col: str = COL_RESTAURANT_LON,
    filename: str = "density_heatmap.png",
) -> str:
    """Plot 2D density heatmap of delivery activity."""
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(df[lon_col], df[lat_col], gridsize=65, cmap="viridis", mincnt=1)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Order Density")
    ax.set_title("Spatial Density Heatmap")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)

    return _save(fig, filename)


def plot_interactive_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    title: str = "Interactive Cluster Map",
    filename: str = "interactive_map.html",
    max_points: int = 3000,
) -> str:
    """Build interactive Folium map with restaurant and delivery points."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except Exception as exc:
        logger.warning("folium unavailable: %s", exc)
        return ""

    data = df.copy()
    data["cluster"] = labels
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)

    center = [float(data[COL_RESTAURANT_LAT].median()), float(data[COL_RESTAURANT_LON].median())]
    fmap = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    rest_layer = MarkerCluster(name="Restaurants").add_to(fmap)
    del_layer = MarkerCluster(name="Delivery Points").add_to(fmap)

    for row in data.itertuples(index=False):
        label = int(row.cluster)
        color = "gray" if label == -1 else PALETTE[label % len(PALETTE)]

        folium.CircleMarker(
            location=[getattr(row, COL_RESTAURANT_LAT), getattr(row, COL_RESTAURANT_LON)],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
        ).add_to(rest_layer)

        folium.CircleMarker(
            location=[getattr(row, COL_DELIVERY_LAT), getattr(row, COL_DELIVERY_LON)],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
        ).add_to(del_layer)

    folium.LayerControl().add_to(fmap)
    folium.map.Marker(
        center,
        icon=folium.DivIcon(html=f"<div style='font-size:14px;font-weight:700'>{title}</div>"),
    ).add_to(fmap)

    out = _ensure_plot_dir() / filename
    fmap.save(str(out))
    return str(out)


def close_all() -> None:
    """Close matplotlib figures to avoid memory build-up in long runs."""
    plt.close("all")
