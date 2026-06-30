"""Feature engineering for geospatial clustering and downstream business analysis."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    COL_CITY,
    COL_CITY_CODE,
    COL_CITY_ORDER_SHARE,
    COL_DELIVERY_DAY,
    COL_DELIVERY_DOW,
    COL_DELIVERY_HOUR,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_DELIVERY_MONTH,
    COL_DROP_X_KM,
    COL_DROP_Y_KM,
    COL_FESTIVAL,
    COL_FESTIVAL_BINARY,
    COL_ORDER_DATE,
    COL_PICKUP_LAG_MIN,
    COL_REGION_CODE,
    COL_REST_X_KM,
    COL_REST_Y_KM,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_TIME_ORDERED,
    COL_TIME_PICKED,
    COL_TRAFFIC,
    COL_TRAFFIC_CODE,
    COL_ZONE_DENSITY,
    COL_ZONE_ID,
    COL_ZONE_LAT_BIN,
    COL_ZONE_LON_BIN,
    DEFAULT_CLUSTERING_FEATURES,
)
from src.distance import pairwise_delivery_distances, project_wgs84_to_web_mercator

logger = logging.getLogger(__name__)


def build_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a leakage-safe feature table for clustering workflows."""
    _add_distance_features(df)
    _add_temporal_features(df)
    _add_pickup_lag_features(df)
    _add_encoded_context_features(df)
    _add_spatial_density_features(df)
    _add_regional_indicators(df)
    _add_projected_coordinates(df)

    return df


def build_downstream_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for supervised downstream benchmarking.

    This function intentionally excludes target columns and target-derived fallbacks.
    """
    return build_clustering_features(df)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for previous notebooks and scripts."""
    return build_clustering_features(df)


def _add_distance_features(df: pd.DataFrame) -> None:
    """Compute distance-related features using geodesic defaults."""
    pairwise_delivery_distances(df, use_geodesic=True)


def _add_temporal_features(df: pd.DataFrame) -> None:
    """Extract hour/day/month/day-of-week signals from date and pickup time."""
    picked = pd.to_datetime(df[COL_TIME_PICKED], format="%H:%M:%S", errors="coerce")
    if picked.isna().all():
        picked = pd.to_datetime(df[COL_TIME_PICKED], format="%H:%M", errors="coerce")

    df[COL_DELIVERY_HOUR] = picked.dt.hour.fillna(12).astype(int)
    df[COL_DELIVERY_DAY] = df[COL_ORDER_DATE].dt.day.fillna(1).astype(int)
    df[COL_DELIVERY_MONTH] = df[COL_ORDER_DATE].dt.month.fillna(1).astype(int)
    df[COL_DELIVERY_DOW] = df[COL_ORDER_DATE].dt.dayofweek.fillna(0).astype(int)


def _add_pickup_lag_features(df: pd.DataFrame) -> None:
    """Estimate order-to-pickup lag with safe midnight rollover handling."""
    ordered = pd.to_datetime(df[COL_TIME_ORDERED], format="%H:%M:%S", errors="coerce")
    picked = pd.to_datetime(df[COL_TIME_PICKED], format="%H:%M:%S", errors="coerce")

    lag = (picked - ordered).dt.total_seconds() / 60.0
    # If pickup clock passes midnight, add one day.
    lag = lag.mask(lag < 0, lag + 24 * 60)

    fallback = lag.median(skipna=True)
    if pd.isna(fallback) or fallback <= 0:
        fallback = 20.0

    df[COL_PICKUP_LAG_MIN] = lag.fillna(float(fallback)).clip(lower=1.0, upper=300.0)


def _add_encoded_context_features(df: pd.DataFrame) -> None:
    """Encode business-context categories into model-friendly numerics."""
    festival_map = {"yes": 1, "no": 0}
    traffic_map = {"low": 1, "medium": 2, "high": 3, "jam": 4}
    city_map = {"semi-urban": 1, "urban": 2, "metropolitian": 3}

    df[COL_FESTIVAL_BINARY] = (
        df[COL_FESTIVAL].astype(str).str.lower().map(festival_map).fillna(0).astype(int)
    )
    df[COL_TRAFFIC_CODE] = (
        df[COL_TRAFFIC].astype(str).str.lower().map(traffic_map).fillna(0).astype(int)
    )
    df[COL_CITY_CODE] = df[COL_CITY].astype(str).str.lower().map(city_map).fillna(0).astype(int)


def _add_spatial_density_features(df: pd.DataFrame) -> None:
    """Create grid zones and local demand density indicators."""
    # 0.1-degree grid (~11 km latitude resolution) to form business micro-zones.
    df[COL_ZONE_LAT_BIN] = (df[COL_RESTAURANT_LAT] / 0.1).round(0) * 0.1
    df[COL_ZONE_LON_BIN] = (df[COL_RESTAURANT_LON] / 0.1).round(0) * 0.1
    df[COL_ZONE_ID] = (
        df[COL_ZONE_LAT_BIN].round(1).astype(str)
        + "_"
        + df[COL_ZONE_LON_BIN].round(1).astype(str)
    )

    zone_counts = df[COL_ZONE_ID].value_counts()
    city_counts = df[COL_CITY].value_counts(dropna=False)

    df[COL_ZONE_DENSITY] = df[COL_ZONE_ID].map(zone_counts).astype(float)
    df[COL_CITY_ORDER_SHARE] = df[COL_CITY].map(city_counts).astype(float) / max(len(df), 1)


def _add_regional_indicators(df: pd.DataFrame) -> None:
    """Encode broad geography regions for cluster explainability."""
    lat_q1, lat_q3 = df[COL_RESTAURANT_LAT].quantile([0.25, 0.75])
    lon_q1, lon_q3 = df[COL_RESTAURANT_LON].quantile([0.25, 0.75])

    # region_code categories:
    # 1=north, 2=south, 3=east, 4=west, 5=central
    region_code = np.full(len(df), 5, dtype=int)
    region_code[df[COL_RESTAURANT_LAT] >= lat_q3] = 1
    region_code[df[COL_RESTAURANT_LAT] <= lat_q1] = 2
    region_code[df[COL_RESTAURANT_LON] >= lon_q3] = 3
    region_code[df[COL_RESTAURANT_LON] <= lon_q1] = 4
    df[COL_REGION_CODE] = region_code


def _add_projected_coordinates(df: pd.DataFrame) -> None:
    """Project lat/lon to EPSG:3857 and store kilometer-scale coordinates."""
    rest_x_m, rest_y_m = project_wgs84_to_web_mercator(
        df[COL_RESTAURANT_LAT].to_numpy(dtype=float),
        df[COL_RESTAURANT_LON].to_numpy(dtype=float),
    )
    drop_x_m, drop_y_m = project_wgs84_to_web_mercator(
        df[COL_DELIVERY_LAT].to_numpy(dtype=float),
        df[COL_DELIVERY_LON].to_numpy(dtype=float),
    )

    df[COL_REST_X_KM] = rest_x_m / 1000.0
    df[COL_REST_Y_KM] = rest_y_m / 1000.0
    df[COL_DROP_X_KM] = drop_x_m / 1000.0
    df[COL_DROP_Y_KM] = drop_y_m / 1000.0


CLUSTERING_FEATURES: list[str] = list(DEFAULT_CLUSTERING_FEATURES)


def select_features(df: pd.DataFrame, features: Optional[list[str]] = None) -> np.ndarray:
    """Select numeric feature matrix for clustering."""
    feature_list = features or CLUSTERING_FEATURES
    missing = [column for column in feature_list if column not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    matrix = df[feature_list].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_matrix = matrix.to_numpy(dtype=float)

    logger.info("Prepared clustering matrix with shape %s", x_matrix.shape)
    return x_matrix
