"""Distance engineering utilities for geospatial analytics.

This module intentionally separates three concepts:
1. Euclidean in degree space (didactic only, not physically valid distance)
2. Euclidean in projected CRS (EPSG:3857), returned in km
3. Earth-surface distance (Haversine / Geodesic), returned in km
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    COL_DELIVERY_DISTANCE,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_EUCLIDEAN_DISTANCE,
    COL_GEODESIC_DISTANCE,
    COL_HAVERSINE_DISTANCE,
    COL_PICKUP_DISTANCE,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
)

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM: float = 6371.0088
EUCLIDEAN_PROJECTED_KM_COL: str = "euclidean_projected_km"


@dataclass
class DistanceComparisonSummary:
    """Summary metrics comparing Euclidean, Haversine, and geodesic distance."""

    n_samples: int
    mean_euclidean_km: float
    mean_haversine_km: float
    mean_geodesic_km: float
    mean_abs_error_euclidean_vs_geodesic: float
    mean_abs_error_haversine_vs_geodesic: float


def haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute vectorized haversine distance in kilometers."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return EARTH_RADIUS_KM * c


def geodesic_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute accurate geodesic distance in kilometers.

    Falls back to haversine when ``geopy`` is unavailable.
    """
    try:
        from geopy.distance import geodesic  # type: ignore

        return float(geodesic((lat1, lon1), (lat2, lon2)).km)
    except Exception:
        return float(
            haversine_vectorized(
                np.array([lat1]),
                np.array([lon1]),
                np.array([lat2]),
                np.array([lon2]),
            )[0]
        )


def geodesic_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute geodesic distance row-wise in kilometers."""
    return np.array(
        [
            geodesic_distance(a, b, c, d)
            for a, b, c, d in zip(lat1, lon1, lat2, lon2, strict=False)
        ],
        dtype=float,
    )


def euclidean_degrees(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance in raw degree-space (didactic)."""
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


def project_wgs84_to_web_mercator(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project EPSG:4326 coordinates to EPSG:3857 in meters."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(longitudes, latitudes)
    return np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)


def euclidean_projected_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance in kilometers after projection to EPSG:3857."""
    x1, y1 = project_wgs84_to_web_mercator(lat1, lon1)
    x2, y2 = project_wgs84_to_web_mercator(lat2, lon2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 1000.0


def pairwise_delivery_distances(
    df: pd.DataFrame,
    *,
    use_geodesic: bool = True,
) -> pd.DataFrame:
    """Compute delivery distance columns and mutate dataframe in-place."""
    lat_r = df[COL_RESTAURANT_LAT].to_numpy(dtype=float)
    lon_r = df[COL_RESTAURANT_LON].to_numpy(dtype=float)
    lat_d = df[COL_DELIVERY_LAT].to_numpy(dtype=float)
    lon_d = df[COL_DELIVERY_LON].to_numpy(dtype=float)

    haversine_dist = haversine_vectorized(lat_r, lon_r, lat_d, lon_d)
    geodesic_dist = geodesic_vectorized(lat_r, lon_r, lat_d, lon_d) if use_geodesic else haversine_dist

    df[COL_HAVERSINE_DISTANCE] = haversine_dist
    # Keep legacy degree-space Euclidean column for educational comparisons.
    df[COL_EUCLIDEAN_DISTANCE] = euclidean_degrees(lat_r, lon_r, lat_d, lon_d)
    df[EUCLIDEAN_PROJECTED_KM_COL] = euclidean_projected_km(lat_r, lon_r, lat_d, lon_d)
    df[COL_GEODESIC_DISTANCE] = geodesic_dist
    df[COL_DELIVERY_DISTANCE] = geodesic_dist
    df[COL_PICKUP_DISTANCE] = 0.0

    logger.info(
        "Distance features computed. mean_km=%.2f max_km=%.2f",
        float(np.nanmean(df[COL_DELIVERY_DISTANCE])),
        float(np.nanmax(df[COL_DELIVERY_DISTANCE])),
    )
    return df


def compare_distance_methods(
    df: pd.DataFrame,
    *,
    sample_size: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, DistanceComparisonSummary]:
    """Create side-by-side distance comparison dataframe and aggregate summary."""
    if len(df) == 0:
        empty = pd.DataFrame(
            columns=[
                COL_EUCLIDEAN_DISTANCE,
                EUCLIDEAN_PROJECTED_KM_COL,
                COL_HAVERSINE_DISTANCE,
                COL_GEODESIC_DISTANCE,
                "abs_error_euclidean_vs_geodesic",
                "abs_error_haversine_vs_geodesic",
            ]
        )
        summary = DistanceComparisonSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return empty, summary

    sample_df = (
        df.sample(n=min(sample_size, len(df)), random_state=random_state)
        .reset_index(drop=True)
        .copy()
    )

    lat_r = sample_df[COL_RESTAURANT_LAT].to_numpy(dtype=float)
    lon_r = sample_df[COL_RESTAURANT_LON].to_numpy(dtype=float)
    lat_d = sample_df[COL_DELIVERY_LAT].to_numpy(dtype=float)
    lon_d = sample_df[COL_DELIVERY_LON].to_numpy(dtype=float)

    eu_deg = euclidean_degrees(lat_r, lon_r, lat_d, lon_d)
    eu_km = euclidean_projected_km(lat_r, lon_r, lat_d, lon_d)
    hav = haversine_vectorized(lat_r, lon_r, lat_d, lon_d)
    geo = geodesic_vectorized(lat_r, lon_r, lat_d, lon_d)

    result = sample_df[
        [COL_RESTAURANT_LAT, COL_RESTAURANT_LON, COL_DELIVERY_LAT, COL_DELIVERY_LON]
    ].copy()
    result[COL_EUCLIDEAN_DISTANCE] = eu_deg
    result[EUCLIDEAN_PROJECTED_KM_COL] = eu_km
    result[COL_HAVERSINE_DISTANCE] = hav
    result[COL_GEODESIC_DISTANCE] = geo
    result["abs_error_euclidean_vs_geodesic"] = np.abs(eu_km - geo)
    result["abs_error_haversine_vs_geodesic"] = np.abs(hav - geo)

    summary = DistanceComparisonSummary(
        n_samples=len(result),
        mean_euclidean_km=float(result[EUCLIDEAN_PROJECTED_KM_COL].mean()),
        mean_haversine_km=float(result[COL_HAVERSINE_DISTANCE].mean()),
        mean_geodesic_km=float(result[COL_GEODESIC_DISTANCE].mean()),
        mean_abs_error_euclidean_vs_geodesic=float(
            result["abs_error_euclidean_vs_geodesic"].mean()
        ),
        mean_abs_error_haversine_vs_geodesic=float(
            result["abs_error_haversine_vs_geodesic"].mean()
        ),
    )

    return result, summary
