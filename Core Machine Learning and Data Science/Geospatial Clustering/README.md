# Geospatial Clustering with Python

Production-grade geospatial analytics project for delivery operations.

## Executive Summary
This project turns raw food-delivery coordinates into operational business zones using robust spatial feature engineering, anomaly detection, and multi-algorithm clustering. It includes educational notebooks, reusable Python modules, a Streamlit demo, and reproducible local execution.

## Business Motivation
Delivery networks need data-driven answers to:
- Where demand is concentrated.
- Which service regions are inefficient or sparse.
- How to redesign zones for better coverage and lower distance cost.

This repository provides an end-to-end workflow from raw data to executive-ready zone insights.

## Dataset Description
- Source: Kaggle `gauravmalik26/food-delivery-dataset`
- File: `train.csv`
- Domain: delivery operations, rider metadata, restaurant/drop coordinates, traffic/weather context, and delivery time outcome.

### Acquisition
Preferred route is Kaggle MCP signed URL:
```bash
KAGGLE_MCP_SIGNED_URL="<signed-url>" uv run python scripts/download_dataset.py
```

Fallback route in script uses Kaggle CLI automatically.

Provenance is written to:
- `outputs/reports/dataset_provenance.json`

## Geospatial Theory
### Coordinate System and CRS
- Raw coordinates are geographic latitude/longitude in `EPSG:4326`.
- Euclidean geometry is performed only after projection to `EPSG:3857`.
- Earth-surface routing distances use Haversine/Geodesic distances in kilometers.

### Distance Metrics
1. Euclidean in degree space (didactic, not physically valid).
2. Euclidean in projected CRS (approximation).
3. Haversine great-circle distance.
4. Geodesic (WGS-84 ellipsoid, highest physical fidelity).

Known-coordinate sanity checks are embedded in `src/geospatial_validation.py`.

## Clustering Algorithms
Implemented and compared:
1. K-Means
2. MiniBatch K-Means
3. DBSCAN (haversine metric on geo coordinates)
4. HDBSCAN (haversine metric on geo coordinates)
5. Agglomerative Clustering
6. Gaussian Mixture Models

### Selection Policy
Model ranking uses a composite score over:
- Silhouette (higher better)
- Davies-Bouldin (lower better)
- Calinski-Harabasz (higher better)
- Stability (ARI across repeated runs)
- Noise ratio (lower better)

## Feature Engineering
### Clustering Feature Set
- Projected coordinate features (`EPSG:3857` in km)
- Geodesic delivery distance
- Temporal context (hour/day/month/day-of-week)
- Pickup lag proxy (order-to-pickup)
- Traffic, city, festival encodings
- Zone density and regional indicators

### Leakage Controls
- Downstream supervised benchmark excludes target and cluster-derived columns.
- Preprocessing is fit on train split only.
- Target-dependent fallback engineering has been removed.

## Outlier and Spatial Analytics
### Outlier Detection
Combined detectors:
- Coordinate bounds
- GPS error logic
- IQR detector
- Mahalanobis distance
- Isolation Forest
- Local Outlier Factor
- DBSCAN noise

Consensus voting identifies robust outliers.

### Advanced Spatial Analytics
- Grid density analysis
- KDE hotspot mining
- Service coverage diagnostics by cluster
- Interactive and static map artifacts

## Business Zone Generation
Clusters are translated into operational zone labels:
- High Demand Zone
- Central Delivery Zone
- Southern Delivery Zone
- Emerging Growth Zone
- Low Activity Zone
- Balanced Service Zone

Each label is attached with deterministic rule logic and human-readable reason text.

## Evaluation and Results
### Cluster Quality Reports
- `outputs/reports/clustering_evaluation.csv`
- `outputs/reports/k_selection_diagnostics.csv`

### Zone and Spatial Reports
- `outputs/reports/business_zone_kpis.csv`
- `outputs/reports/grid_density.csv`
- `outputs/reports/kde_hotspots.csv`
- `outputs/reports/spatial_density_summary.csv`

### AutoML Benchmark
- Manual ML vs LazyPredict vs FLAML vs PyCaret
- Report: `outputs/reports/downstream_benchmark.csv`

### Validated Snapshot (2026-06-24)
- Raw rows: `45,593`; cleaned rows: `40,007`; removed as invalid: `5,586`.
- Detected outliers (consensus): `1,503` rows (`3.76%` of cleaned set).
- Best clustering model (composite): `HDBSCAN` with `241` clusters and `0.64%` noise points.
- Best downstream RMSE (delivery-time prediction): `3.87` (manual RandomForest in pipeline run).
- Geospatial checks: all distance and CRS validation cases passed (`outputs/reports/geospatial_validation.json`).
- End-to-end verification (`ruff + pytest + pipeline + notebooks`) passed:
  - `outputs/reports/verification_summary.json`

## Streamlit Demo
Launch local app:
```bash
uv run streamlit run streamlit_app/app.py
```

The app supports:
- CSV upload with schema/coordinate validation
- clustering execution
- outlier and zone analysis
- spatial hotspot views
- benchmark execution
- report/download surfaces

## Notebook Mini-Book
Canonical chapter sequence:
1. `01_geospatial_foundations.ipynb`
2. `02_dataset_profile_and_eda.ipynb`
3. `03_distance_engineering.ipynb`
4. `04_clustering_algorithms_and_k_selection.ipynb`
5. `05_outliers_business_zones_and_interpretability.ipynb`
6. `06_advanced_spatial_analysis.ipynb`
7. `07_downstream_automl_benchmark.ipynb`
8. `08_end_to_end_pipeline_and_streamlit_demo.ipynb`

Regenerate notebooks:
```bash
uv run python notebooks/generate_notebooks.py
```

## Local Setup and Reproducibility
### Runtime
- Python `3.11.9` (pinned for stable `pycaret` compatibility)
- `uv` package management

### Environment
```bash
uv venv .venv --python 3.11.9
source .venv/bin/activate
uv sync --extra dev
```

### Full Pipeline
```bash
uv run python scripts/run_pipeline.py \
  --algorithms kmeans minibatch_kmeans dbscan hdbscan agglomerative gmm \
  --run-automl
```

### Notebook Execution
```bash
uv run python scripts/run_notebooks.py
```

### One-command Verification
```bash
uv run python scripts/verify_project.py
```

## Key Artifacts
- Pipeline summary: `outputs/pipeline_report.json`
- Geospatial validation: `outputs/reports/geospatial_validation.json`
- Dataset quality profile: `outputs/reports/dataset_profile.json`
- Cluster assignments: `data/processed/cluster_assignments.csv`
- Interactive map: `outputs/plots/interactive_map.html`
- Verification summary: `outputs/reports/verification_summary.json`

## Lessons Learned
- Geospatial data quality dominates model quality.
- CRS discipline is mandatory for mathematically valid spatial analysis.
- Density-based methods need geospatial distance metrics (haversine/geodesic), not default Euclidean assumptions.
- Cluster quality must be judged with multiple metrics and stability checks, not silhouette alone.

## Future Improvements
1. Add road-network distance/time enrichment via routing engines.
2. Add temporal drift monitoring for monthly zone refresh.
3. Add constrained clustering for SLA-aware balancing.
4. Add experiment tracking (MLflow/W&B) for benchmarking history.
