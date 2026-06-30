"""Streamlit demo for geospatial clustering and business-zone analytics."""

from __future__ import annotations

import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business_zones import (  # noqa: E402
    assign_business_zone_labels,
    attach_zone_labels,
    build_cluster_kpis,
)
from src.clustering import run_algorithm  # noqa: E402
from src.config import (  # noqa: E402
    CLUSTERING_ALGORITHMS,
    COL_CLUSTER,
    COL_DELIVERY_DISTANCE,
    COL_PICKUP_LAG_MIN,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_ZONE_LABEL,
    TRAIN_FILE_PATH,
)
from src.data_loader import load_and_clean_data, load_raw_data  # noqa: E402
from src.data_quality import run_quality_gate  # noqa: E402
from src.downstream_benchmark import run_full_benchmark  # noqa: E402
from src.features import (  # noqa: E402
    build_clustering_features,
    build_downstream_features,
    select_features,
)
from src.outlier_detection import detect_outliers  # noqa: E402
from src.pipeline import GeospatialClusteringPipeline  # noqa: E402
from src.spatial_analysis import (  # noqa: E402
    build_grid_density,
    kde_hotspots,
    service_coverage_analysis,
)
from src.streamlit_validation import validate_upload_dataframe  # noqa: E402
from src.visualization import plot_interactive_map  # noqa: E402

st.set_page_config(page_title="Geospatial Clustering Studio", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    return load_raw_data(TRAIN_FILE_PATH)


def _save_temp_csv(df: pd.DataFrame) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        return Path(tmp.name)


def _load_uploaded_dataframe(uploaded_file) -> tuple[pd.DataFrame | None, list[str]]:
    if uploaded_file is None:
        return None, ["No file uploaded."]

    try:
        content = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        return None, ["Uploaded file is not valid UTF-8 text."]

    try:
        df = pd.read_csv(StringIO(content), low_memory=False)
    except Exception as exc:  # pragma: no cover - runtime guard
        return None, [f"Could not parse CSV: {exc}"]

    return df, validate_upload_dataframe(df)


def _clean_features_from_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    tmp_path = _save_temp_csv(df)
    clean_df = load_and_clean_data(path=tmp_path, validate=True)
    feat_df = build_clustering_features(clean_df.copy())
    x_matrix = select_features(feat_df)
    return feat_df, x_matrix


st.title("Geospatial Clustering with Python")
st.caption("Location Intelligence, Spatial Analytics, and Business Zone Optimization")

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("Upload CSV (optional)", type=["csv"])
    use_default = st.checkbox("Use project default dataset", value=upload is None)

    if use_default:
        raw_df = load_default_dataset()
        upload_errors: list[str] = []
    else:
        uploaded_df, upload_errors = _load_uploaded_dataframe(upload)
        raw_df = uploaded_df if uploaded_df is not None else pd.DataFrame()

    if upload_errors and not use_default:
        for error in upload_errors:
            st.error(error)

    st.write(f"Raw rows: {len(raw_df):,}")

    page = st.radio(
        "Page",
        [
            "Dataset Profile",
            "Clustering Lab",
            "Outlier + Zones",
            "Spatial Analytics",
            "Downstream AutoML",
            "Pipeline Runner",
        ],
    )

if not use_default and (raw_df.empty or upload_errors):
    st.warning("Upload a valid CSV with required schema to continue.")
    st.stop()

if page == "Dataset Profile":
    st.subheader("Dataset Structure and Quality Gate")
    quality = run_quality_gate(raw_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{quality.n_rows:,}")
    c2.metric("Columns", f"{quality.n_columns}")
    c3.metric("Out-of-India rows", f"{quality.rows_outside_india_bounds:,}")

    st.markdown("### Missing Values (%)")
    missing_df = pd.DataFrame(
        {
            "column": list(quality.missing_by_column_pct.keys()),
            "missing_pct": list(quality.missing_by_column_pct.values()),
        }
    ).sort_values("missing_pct", ascending=False)
    st.dataframe(missing_df, use_container_width=True)

    st.markdown("### Coordinate Ranges")
    st.json(quality.coordinate_ranges)

    if quality.warnings:
        st.markdown("### Quality Warnings")
        for warning in quality.warnings:
            st.warning(warning)

    st.markdown("### Data Sample")
    st.dataframe(raw_df.head(100), use_container_width=True)

elif page == "Clustering Lab":
    st.subheader("Interactive Clustering")
    feat_df, x_matrix = _clean_features_from_dataframe(raw_df)

    algo = st.selectbox("Algorithm", CLUSTERING_ALGORITHMS)
    keep_outliers = st.checkbox("Keep outliers in clustering", value=False)

    if st.button("Run Clustering", type="primary"):
        geo_coords = feat_df[[COL_RESTAURANT_LAT, COL_RESTAURANT_LON]].to_numpy(dtype=float)
        consensus, _ = detect_outliers(feat_df, feature_matrix=x_matrix)
        if keep_outliers:
            work_df = feat_df.copy()
            x_work = x_matrix
            geo_work = geo_coords
        else:
            mask = ~consensus.mask
            work_df = feat_df.loc[mask].reset_index(drop=True)
            x_work = x_matrix[mask]
            geo_work = geo_coords[mask]

        result = run_algorithm(algo, x_work, geo_coords=geo_work)
        temp = work_df.copy()
        temp[COL_CLUSTER] = result.labels

        st.write(f"Clusters: {result.n_clusters} | Noise points: {result.n_noise}")

        fig = px.scatter_map(
            temp,
            lat=COL_RESTAURANT_LAT,
            lon=COL_RESTAURANT_LON,
            color=temp[COL_CLUSTER].astype(str),
            hover_data=[COL_DELIVERY_DISTANCE, COL_PICKUP_LAG_MIN],
            zoom=4,
            title=f"Geospatial Clusters ({algo})",
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

        map_path = plot_interactive_map(temp, result.labels, filename=f"streamlit_map_{algo}.html")
        if map_path:
            st.success(f"Interactive map saved: {map_path}")

        csv_bytes = temp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Clustered Data",
            data=csv_bytes,
            file_name="clustered_output.csv",
            mime="text/csv",
        )

elif page == "Outlier + Zones":
    st.subheader("Outlier Detection and Business Zone Creation")
    feat_df, x_matrix = _clean_features_from_dataframe(raw_df)

    consensus, reports = detect_outliers(feat_df, feature_matrix=x_matrix)
    st.markdown("### Outlier Counts")
    report_df = pd.DataFrame({"method": [r.name for r in reports], "outliers": [r.n_outliers for r in reports]})
    st.dataframe(report_df, use_container_width=True)

    mask = ~consensus.mask
    feat_clean = feat_df.loc[mask].reset_index(drop=True)
    x_clean = x_matrix[mask]
    geo_clean = feat_clean[[COL_RESTAURANT_LAT, COL_RESTAURANT_LON]].to_numpy(dtype=float)

    result = run_algorithm("kmeans", x_clean, geo_coords=geo_clean)
    kpis = build_cluster_kpis(feat_clean, result.labels)
    zone_kpis, rule_summary = assign_business_zone_labels(kpis)
    labeled = attach_zone_labels(feat_clean, result.labels, zone_kpis)

    st.markdown("### Business Zone KPIs")
    st.dataframe(zone_kpis, use_container_width=True)
    st.json(rule_summary.__dict__)

    fig_zone = px.scatter_map(
        labeled,
        lat=COL_RESTAURANT_LAT,
        lon=COL_RESTAURANT_LON,
        color=COL_ZONE_LABEL,
        hover_data=[COL_CLUSTER, COL_DELIVERY_DISTANCE, COL_PICKUP_LAG_MIN],
        zoom=4,
        title="Business Zone Map",
        height=650,
    )
    st.plotly_chart(fig_zone, use_container_width=True)

elif page == "Spatial Analytics":
    st.subheader("Heatmaps, Hotspots, and Service Coverage")
    feat_df, x_matrix = _clean_features_from_dataframe(raw_df)
    geo_coords = feat_df[[COL_RESTAURANT_LAT, COL_RESTAURANT_LON]].to_numpy(dtype=float)
    result = run_algorithm("kmeans", x_matrix, geo_coords=geo_coords)

    density_df = build_grid_density(feat_df)
    hotspots_df = kde_hotspots(feat_df, top_n=120)
    coverage_df = service_coverage_analysis(feat_df, result.labels)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Hotspot Candidates (Top KDE)")
        st.dataframe(hotspots_df.head(30), use_container_width=True)
    with c2:
        st.markdown("### Service Coverage by Cluster")
        st.dataframe(coverage_df, use_container_width=True)

    heat = px.density_map(
        feat_df,
        lat=COL_RESTAURANT_LAT,
        lon=COL_RESTAURANT_LON,
        radius=10,
        zoom=4,
        title="Demand Density Heatmap",
        height=650,
    )
    st.plotly_chart(heat, use_container_width=True)

    st.markdown("### Grid Density Table")
    st.dataframe(density_df.head(50), use_container_width=True)

elif page == "Downstream AutoML":
    st.subheader("Predict Delivery Time: Manual ML vs AutoML")
    tmp_path = _save_temp_csv(raw_df)
    clean_df = load_and_clean_data(path=tmp_path, validate=True)
    feat_df = build_downstream_features(clean_df.copy())

    if st.button("Run AutoML Benchmark", type="primary"):
        with st.spinner("Benchmarking models..."):
            bench_df = run_full_benchmark(feat_df)
        st.dataframe(bench_df.sort_values(["status", "rmse"]), use_container_width=True)

        ok_df = bench_df[bench_df["status"] == "ok"]
        if not ok_df.empty:
            best = ok_df.sort_values("rmse").iloc[0]
            st.success(f"Best model: {best['framework']} / {best['model']} | RMSE={best['rmse']:.3f}")

elif page == "Pipeline Runner":
    st.subheader("One-Click End-to-End Pipeline")
    run_automl = st.checkbox("Include downstream AutoML benchmark", value=False)

    if st.button("Run Full Pipeline", type="primary"):
        with st.spinner("Executing end-to-end pipeline..."):
            tmp_path = _save_temp_csv(raw_df)
            pipeline = GeospatialClusteringPipeline(
                data_path=tmp_path,
                remove_outliers=True,
                algorithms=list(CLUSTERING_ALGORITHMS),
                run_downstream_automl=run_automl,
            )
            report = pipeline.run()

        st.success("Pipeline finished")
        st.write(f"Best algorithm: **{report.best_algorithm}**")
        st.write("Business impact:")
        st.json(report.business_impact)

        if report.warnings:
            st.markdown("### Pipeline Warnings")
            for warning in report.warnings:
                st.warning(warning)

        artifact_df = pd.DataFrame({"artifact": report.artifact_paths.keys(), "path": report.artifact_paths.values()})
        report_df = pd.DataFrame({"report": report.report_paths.keys(), "path": report.report_paths.values()})

        st.markdown("### Artifact Paths")
        st.dataframe(artifact_df, use_container_width=True)
        st.markdown("### Report Paths")
        st.dataframe(report_df, use_container_width=True)

        json_bytes = json.dumps(report.__dict__, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download Pipeline Report JSON",
            data=json_bytes,
            file_name="pipeline_report.json",
            mime="application/json",
        )

st.sidebar.markdown("---")
st.sidebar.caption("Portfolio Demo: Geospatial Clustering with Python")
