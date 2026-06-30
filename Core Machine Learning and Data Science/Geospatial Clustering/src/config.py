"""Central configuration for the geospatial clustering project.

This module defines paths, schema contracts, feature names, model defaults, and
geographic constraints. Keeping these values in one place makes the project easy
to maintain and safer to refactor.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
PLOTS_DIR: Final[Path] = OUTPUTS_DIR / "plots"
REPORTS_DIR: Final[Path] = OUTPUTS_DIR / "reports"
MAPS_DIR: Final[Path] = OUTPUTS_DIR / "maps"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
STREAMLIT_DIR: Final[Path] = PROJECT_ROOT / "streamlit_app"

for _path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, MAPS_DIR]:
    _path.mkdir(parents=True, exist_ok=True)

# Backward-compatible alias used by older modules.
OUTPUT_DIR: Final[Path] = OUTPUTS_DIR

# Backward compatibility aliases
DATA_PATH: Final[str] = str(RAW_DATA_DIR)
PROCESSED_PATH: Final[str] = str(PROCESSED_DATA_DIR)
OUTPUTS_PATH: Final[str] = str(OUTPUTS_DIR)
MAPS_PATH: Final[str] = str(MAPS_DIR)
FIGURES_PATH: Final[str] = str(PLOTS_DIR)
REPORTS_PATH: Final[str] = str(REPORTS_DIR)
PROCESSED_FILENAME: Final[str] = "processed_data.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
KAGGLE_OWNER: Final[str] = "gauravmalik26"
KAGGLE_DATASET_SLUG: Final[str] = "food-delivery-dataset"
TRAIN_FILENAME: Final[str] = "train.csv"
TEST_FILENAME: Final[str] = "test.csv"
SAMPLE_SUBMISSION_FILENAME: Final[str] = "Sample_Submission.csv"
TRAIN_FILE_PATH: Final[Path] = RAW_DATA_DIR / TRAIN_FILENAME

# ---------------------------------------------------------------------------
# Raw schema columns
# ---------------------------------------------------------------------------
COL_ID: Final[str] = "ID"
COL_DELIVERY_PERSON_ID: Final[str] = "Delivery_person_ID"
COL_AGE: Final[str] = "Delivery_person_Age"
COL_RATINGS: Final[str] = "Delivery_person_Ratings"
COL_RESTAURANT_LAT: Final[str] = "Restaurant_latitude"
COL_RESTAURANT_LON: Final[str] = "Restaurant_longitude"
COL_DELIVERY_LAT: Final[str] = "Delivery_location_latitude"
COL_DELIVERY_LON: Final[str] = "Delivery_location_longitude"
COL_ORDER_DATE: Final[str] = "Order_Date"
COL_TIME_ORDERED: Final[str] = "Time_Orderd"
COL_TIME_PICKED: Final[str] = "Time_Order_picked"
COL_WEATHER: Final[str] = "Weatherconditions"
COL_TRAFFIC: Final[str] = "Road_traffic_density"
COL_VEHICLE_COND: Final[str] = "Vehicle_condition"
COL_ORDER_TYPE: Final[str] = "Type_of_order"
COL_VEHICLE_TYPE: Final[str] = "Type_of_vehicle"
COL_MULTI_DELIVERY: Final[str] = "multiple_deliveries"
COL_FESTIVAL: Final[str] = "Festival"
COL_CITY: Final[str] = "City"
COL_TIME_TAKEN: Final[str] = "Time_taken(min)"

REQUIRED_COLUMNS: Final[list[str]] = [
    COL_ID,
    COL_AGE,
    COL_RATINGS,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_ORDER_DATE,
    COL_TIME_ORDERED,
    COL_TIME_PICKED,
    COL_TRAFFIC,
    COL_CITY,
    COL_TIME_TAKEN,
]

COORDINATE_COLUMNS: Final[list[str]] = [
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
]

# ---------------------------------------------------------------------------
# Engineered feature columns
# ---------------------------------------------------------------------------
COL_DELIVERY_DISTANCE: Final[str] = "delivery_distance_km"
COL_PICKUP_DISTANCE: Final[str] = "pickup_distance_km"
COL_EUCLIDEAN_DISTANCE: Final[str] = "euclidean_distance_deg"
COL_GEODESIC_DISTANCE: Final[str] = "geodesic_distance_km"
COL_HAVERSINE_DISTANCE: Final[str] = "haversine_distance_km"
COL_DELIVERY_HOUR: Final[str] = "delivery_hour"
COL_DELIVERY_DAY: Final[str] = "delivery_day"
COL_DELIVERY_MONTH: Final[str] = "delivery_month"
COL_DELIVERY_DOW: Final[str] = "delivery_dayofweek"
COL_DURATION_MIN: Final[str] = "duration_min"
COL_PICKUP_LAG_MIN: Final[str] = "pickup_lag_min"
COL_SPEED_KMPH: Final[str] = "speed_kmph"
COL_FESTIVAL_BINARY: Final[str] = "festival_binary"
COL_TRAFFIC_CODE: Final[str] = "traffic_code"
COL_CITY_CODE: Final[str] = "city_code"
COL_ZONE_LAT_BIN: Final[str] = "zone_lat_bin"
COL_ZONE_LON_BIN: Final[str] = "zone_lon_bin"
COL_ZONE_ID: Final[str] = "zone_id"
COL_ZONE_DENSITY: Final[str] = "zone_order_density"
COL_CITY_ORDER_SHARE: Final[str] = "city_order_share"
COL_REGION_CODE: Final[str] = "region_code"
COL_REST_X_KM: Final[str] = "restaurant_x_km"
COL_REST_Y_KM: Final[str] = "restaurant_y_km"
COL_DROP_X_KM: Final[str] = "delivery_x_km"
COL_DROP_Y_KM: Final[str] = "delivery_y_km"
COL_CLUSTER: Final[str] = "cluster"
COL_ZONE_LABEL: Final[str] = "zone_label"
COL_OUTLIER: Final[str] = "is_outlier"
COL_OUTLIER_SCORE: Final[str] = "outlier_vote_count"

# ---------------------------------------------------------------------------
# Geography / coordinate constraints
# ---------------------------------------------------------------------------
INDIA_BOUNDS: Final[dict[str, float]] = {
    "lat_min": 6.0,
    "lat_max": 37.0,
    "lon_min": 68.0,
    "lon_max": 98.0,
}
INDIA_LAT_MIN: Final[float] = INDIA_BOUNDS["lat_min"]
INDIA_LAT_MAX: Final[float] = INDIA_BOUNDS["lat_max"]
INDIA_LON_MIN: Final[float] = INDIA_BOUNDS["lon_min"]
INDIA_LON_MAX: Final[float] = INDIA_BOUNDS["lon_max"]

# Approximate conversion constants for small-area calculations.
KM_PER_DEG_LAT: Final[float] = 111.32

# ---------------------------------------------------------------------------
# Clustering settings
# ---------------------------------------------------------------------------
CLUSTERING_ALGORITHMS: Final[list[str]] = [
    "kmeans",
    "minibatch_kmeans",
    "dbscan",
    "hdbscan",
    "agglomerative",
    "gmm",
]

CLUSTERING_DEFAULTS: Final[dict[str, dict]] = {
    "kmeans": {"n_clusters": 6, "random_state": RANDOM_SEED, "n_init": "auto"},
    "minibatch_kmeans": {
        "n_clusters": 6,
        "batch_size": 2048,
        "random_state": RANDOM_SEED,
        "n_init": "auto",
    },
    "dbscan": {"eps_km": "auto", "min_samples": 30},
    "hdbscan": {"min_cluster_size": 120, "min_samples": 40},
    "agglomerative": {"n_clusters": 6, "linkage": "ward"},
    "gmm": {"n_components": 6, "covariance_type": "full", "random_state": RANDOM_SEED},
}

K_SELECTION_RANGE: Final[range] = range(2, 13)

EVALUATION_METRICS: Final[list[str]] = [
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]

# ---------------------------------------------------------------------------
# Feature matrix defaults
# ---------------------------------------------------------------------------
DEFAULT_CLUSTERING_FEATURES: ClassVar[list[str]] = [
    COL_REST_X_KM,
    COL_REST_Y_KM,
    COL_DROP_X_KM,
    COL_DROP_Y_KM,
    COL_DELIVERY_DISTANCE,
    COL_DELIVERY_HOUR,
    COL_DELIVERY_DAY,
    COL_DELIVERY_MONTH,
    COL_DELIVERY_DOW,
    COL_PICKUP_LAG_MIN,
    COL_FESTIVAL_BINARY,
    COL_TRAFFIC_CODE,
    COL_CITY_CODE,
    COL_VEHICLE_COND,
    COL_MULTI_DELIVERY,
    COL_ZONE_DENSITY,
    COL_CITY_ORDER_SHARE,
    COL_REGION_CODE,
]

# ---------------------------------------------------------------------------
# Report output paths
# ---------------------------------------------------------------------------
PROFILE_REPORT_PATH: Final[Path] = REPORTS_DIR / "dataset_profile.json"
CLUSTER_EVAL_REPORT_PATH: Final[Path] = REPORTS_DIR / "clustering_evaluation.csv"
ZONE_KPI_REPORT_PATH: Final[Path] = REPORTS_DIR / "business_zone_kpis.csv"
SPATIAL_SUMMARY_REPORT_PATH: Final[Path] = REPORTS_DIR / "spatial_density_summary.csv"
PIPELINE_REPORT_PATH: Final[Path] = OUTPUTS_DIR / "pipeline_report.json"
ASSIGNMENTS_PATH: Final[Path] = PROCESSED_DATA_DIR / "cluster_assignments.csv"
BENCHMARK_REPORT_PATH: Final[Path] = REPORTS_DIR / "downstream_benchmark.csv"
GEOSPATIAL_VALIDATION_REPORT_PATH: Final[Path] = REPORTS_DIR / "geospatial_validation.json"

# ---------------------------------------------------------------------------
# Business-zone labels
# ---------------------------------------------------------------------------
BUSINESS_ZONE_LABELS: Final[list[str]] = [
    "High Demand Zone",
    "Central Delivery Zone",
    "Southern Delivery Zone",
    "Emerging Growth Zone",
    "Low Activity Zone",
    "Balanced Service Zone",
]

# ---------------------------------------------------------------------------
# Outlier detection defaults
# ---------------------------------------------------------------------------
OUTLIER_METHODS: Final[list[str]] = [
    "coordinate_bounds",
    "gps_error",
    "iqr",
    "mahalanobis",
    "isolation_forest",
    "lof",
    "dbscan_noise",
]

DEFAULT_OUTLIER_CONTAMINATION: Final[float] = 0.03
