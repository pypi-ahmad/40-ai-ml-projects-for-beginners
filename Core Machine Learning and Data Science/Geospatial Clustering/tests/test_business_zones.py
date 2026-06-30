"""Tests for business-zone labeling logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.business_zones import assign_business_zone_labels, attach_zone_labels, build_cluster_kpis
from src.config import BUSINESS_ZONE_LABELS, COL_CLUSTER, COL_ZONE_LABEL


def test_zone_labels_generated_from_kpis() -> None:
    df = pd.DataFrame(
        {
            "delivery_distance_km": [1.0, 2.0, 10.0, 12.0, 8.0, 9.0],
            "duration_min": [15, 18, 40, 45, 30, 35],
            "speed_kmph": [20, 18, 12, 10, 16, 15],
            "Restaurant_latitude": [19.1, 19.2, 12.9, 13.0, 28.6, 28.7],
            "Restaurant_longitude": [72.8, 72.9, 77.5, 77.6, 77.2, 77.3],
        }
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    kpis = build_cluster_kpis(df, labels)
    labeled_kpis, _ = assign_business_zone_labels(kpis)
    row_level = attach_zone_labels(df, labels, labeled_kpis)

    assert COL_ZONE_LABEL in labeled_kpis.columns
    assert set(labeled_kpis[COL_ZONE_LABEL]).issubset(set(BUSINESS_ZONE_LABELS))
    assert COL_CLUSTER in row_level.columns
    assert COL_ZONE_LABEL in row_level.columns
