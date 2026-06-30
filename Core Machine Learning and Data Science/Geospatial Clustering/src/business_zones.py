"""Business-zone creation and labeling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    BUSINESS_ZONE_LABELS,
    COL_CLUSTER,
    COL_DELIVERY_DISTANCE,
    COL_OUTLIER,
    COL_PICKUP_LAG_MIN,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_ZONE_LABEL,
)

ZONE_REASON_COL = "zone_reason"


@dataclass
class ZoneLabelRuleSummary:
    """Diagnostics for deterministic business-zone labeling rules."""

    high_demand_threshold: float
    low_activity_threshold: float
    high_density_threshold: float
    centrality_threshold_km: float
    southern_lat_threshold: float


def build_cluster_kpis(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    outlier_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute operational KPIs per cluster."""
    temp = df.copy()
    temp[COL_CLUSTER] = labels

    # Keep KPI aggregation robust for external callers that provide
    # precomputed duration fields but not engineered pickup-lag features.
    if COL_PICKUP_LAG_MIN not in temp.columns:
        if "duration_min" in temp.columns:
            temp[COL_PICKUP_LAG_MIN] = pd.to_numeric(temp["duration_min"], errors="coerce")
        elif "Time_taken(min)" in temp.columns:
            temp[COL_PICKUP_LAG_MIN] = pd.to_numeric(temp["Time_taken(min)"], errors="coerce")
        else:
            temp[COL_PICKUP_LAG_MIN] = np.nan

    if outlier_mask is None:
        temp[COL_OUTLIER] = False
    else:
        temp[COL_OUTLIER] = outlier_mask

    summary = (
        temp.groupby(COL_CLUSTER, as_index=False)
        .agg(
            orders=(COL_CLUSTER, "count"),
            avg_distance_km=(COL_DELIVERY_DISTANCE, "mean"),
            avg_pickup_lag_min=(COL_PICKUP_LAG_MIN, "mean"),
            centroid_lat=(COL_RESTAURANT_LAT, "mean"),
            centroid_lon=(COL_RESTAURANT_LON, "mean"),
            outlier_pct=(COL_OUTLIER, lambda values: float(np.mean(values) * 100)),
        )
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )

    total_orders = max(summary["orders"].sum(), 1)
    summary["coverage_pct"] = summary["orders"] / total_orders * 100.0
    summary["density_index"] = summary["orders"] / summary["avg_distance_km"].clip(lower=0.1)

    global_lat = float(temp[COL_RESTAURANT_LAT].mean())
    global_lon = float(temp[COL_RESTAURANT_LON].mean())
    summary["centrality_km"] = np.sqrt(
        ((summary["centroid_lat"] - global_lat) * 111.32) ** 2
        + ((summary["centroid_lon"] - global_lon) * 111.32 * np.cos(np.radians(summary["centroid_lat"]))) ** 2
    )

    return summary


def assign_business_zone_labels(kpi_df: pd.DataFrame) -> tuple[pd.DataFrame, ZoneLabelRuleSummary]:
    """Assign deterministic business labels to cluster KPI table."""
    table = kpi_df.copy()

    high_demand_threshold = float(table["orders"].quantile(0.75))
    low_activity_threshold = float(table["orders"].quantile(0.25))
    high_density_threshold = float(table["density_index"].quantile(0.65))
    centrality_threshold_km = float(table["centrality_km"].quantile(0.35))
    southern_lat_threshold = float(table["centroid_lat"].quantile(0.25))

    labels: list[str] = []
    reasons: list[str] = []
    for _, row in table.iterrows():
        if row["orders"] >= high_demand_threshold and row["density_index"] >= high_density_threshold:
            labels.append("High Demand Zone")
            reasons.append("High order volume with strong density index.")
        elif row["centrality_km"] <= centrality_threshold_km:
            labels.append("Central Delivery Zone")
            reasons.append("Cluster centroid is near network center; good hub candidate.")
        elif row["centroid_lat"] <= southern_lat_threshold:
            labels.append("Southern Delivery Zone")
            reasons.append("Cluster concentrated in southern latitude band.")
        elif row["orders"] <= low_activity_threshold:
            labels.append("Low Activity Zone")
            reasons.append("Low demand volume relative to other zones.")
        else:
            labels.append("Emerging Growth Zone")
            reasons.append("Mid-demand zone with growth potential.")

    labels = [label if label in BUSINESS_ZONE_LABELS else "Balanced Service Zone" for label in labels]
    table[COL_ZONE_LABEL] = labels
    table[ZONE_REASON_COL] = reasons

    rule_summary = ZoneLabelRuleSummary(
        high_demand_threshold=high_demand_threshold,
        low_activity_threshold=low_activity_threshold,
        high_density_threshold=high_density_threshold,
        centrality_threshold_km=centrality_threshold_km,
        southern_lat_threshold=southern_lat_threshold,
    )

    return table, rule_summary


def attach_zone_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    zone_kpi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``cluster`` and ``zone_label`` columns to row-level dataset."""
    mapping = zone_kpi_df.set_index(COL_CLUSTER)[COL_ZONE_LABEL].to_dict()
    reason_map = zone_kpi_df.set_index(COL_CLUSTER)[ZONE_REASON_COL].to_dict()
    result = df.copy()
    result[COL_CLUSTER] = labels
    result[COL_ZONE_LABEL] = result[COL_CLUSTER].map(mapping).fillna("Balanced Service Zone")
    result[ZONE_REASON_COL] = result[COL_CLUSTER].map(reason_map).fillna("No rule matched.")
    return result
