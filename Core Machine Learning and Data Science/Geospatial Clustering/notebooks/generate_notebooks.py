"""Generate the multi-notebook zero-to-hero geospatial mini-book."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def common_setup_cells() -> list:
    return [
        code(
            """
            from __future__ import annotations

            import sys
            import warnings
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            warnings.filterwarnings("ignore")

            if (Path.cwd() / "src").exists():
                PROJECT_ROOT = Path.cwd()
            elif (Path.cwd().parent / "src").exists():
                PROJECT_ROOT = Path.cwd().parent
            else:
                raise RuntimeError("Could not locate project root.")

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            print(f"Project root: {PROJECT_ROOT}")
            """
        ),
        code(
            """
            from src.config import TRAIN_FILE_PATH
            from src.data_loader import load_raw_data, load_and_clean_data, explain_dataset_fields
            """
        ),
    ]


def educational_block() -> list:
    return [
        md("""
        ## Definition
        Explain the concept in one sentence and why it matters in delivery analytics.

        ## Theory
        Describe the assumptions and algorithmic behavior at a practical level.

        ## Mathematical Intuition
        Show the formula or geometry intuition behind the method.

        ## Visual Explanation
        Use a chart/map to make the concept concrete.

        ## Code Explanation
        Walk through what each code block does and what inputs/outputs mean.

        ## Interpretation of Results
        Connect metrics and visuals to operational decisions.
        """),
    ]


def notebook_01() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # 01 - Geospatial Foundations

            Zero-to-hero introduction to spatial data for logistics and location intelligence.

            **Real-world examples**
            - Food delivery route optimization
            - Fleet dispatch balancing
            - Warehouse and dark-store placement
            - Ride-sharing demand zoning
            """
        ),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.distance import compare_distance_methods

            raw = load_raw_data(TRAIN_FILE_PATH)
            comp_df, comp_summary = compare_distance_methods(raw, sample_size=800)
            display(comp_df.head())
            print(comp_summary)
            """
        ),
        code(
            """
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].hist(comp_df["abs_error_euclidean_vs_geodesic"], bins=40, color="#d62728", alpha=0.8)
            ax[0].set_title("Projected Euclidean error vs Geodesic")
            ax[0].set_xlabel("Absolute Error (km)")

            ax[1].hist(comp_df["abs_error_haversine_vs_geodesic"], bins=40, color="#2ca02c", alpha=0.8)
            ax[1].set_title("Haversine error vs Geodesic")
            ax[1].set_xlabel("Absolute Error (km)")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            **Interpretation**
            - Earth-curvature-aware methods (Haversine/Geodesic) are mandatory for delivery distance features.
            - Euclidean geometry should only be used in a projected CRS and even then validated.
            """
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_02() -> nbf.NotebookNode:
    cells = [
        md("""
        # 02 - Dataset Profile and EDA

        Learn how to audit geospatial datasets before any modeling.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.data_quality import run_quality_gate, quality_gate_to_markdown

            raw_df = load_raw_data(TRAIN_FILE_PATH)
            quality = run_quality_gate(raw_df)
            print(quality_gate_to_markdown(quality))

            fields = explain_dataset_fields()
            display(pd.DataFrame({"field": fields.keys(), "explanation": fields.values()}))
            """
        ),
        code(
            """
            clean_df = load_and_clean_data(TRAIN_FILE_PATH)
            print("Raw shape:", raw_df.shape)
            print("Clean shape:", clean_df.shape)
            display(clean_df.head())
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            clean_df["Restaurant_latitude"].hist(ax=axes[0,0], bins=50)
            axes[0,0].set_title("Restaurant Latitude")

            clean_df["Restaurant_longitude"].hist(ax=axes[0,1], bins=50)
            axes[0,1].set_title("Restaurant Longitude")

            clean_df["Delivery_location_latitude"].hist(ax=axes[1,0], bins=50)
            axes[1,0].set_title("Delivery Latitude")

            clean_df["Delivery_location_longitude"].hist(ax=axes[1,1], bins=50)
            axes[1,1].set_title("Delivery Longitude")

            plt.tight_layout()
            plt.show()
            """
        ),
        md("""
        **Interpretation**
        - High anomaly rates in coordinates directly damage cluster quality.
        - Data cleaning is not optional in geospatial pipelines.
        """),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_03() -> nbf.NotebookNode:
    cells = [
        md("""
        # 03 - Distance Engineering

        Compare Euclidean, Haversine, and Geodesic distance and quantify the tradeoffs.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.distance import compare_distance_methods

            raw = load_raw_data(TRAIN_FILE_PATH)
            comparison, summary = compare_distance_methods(raw, sample_size=1200)
            display(comparison.head())
            print(summary)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            comparison["euclidean_projected_km"].hist(ax=axes[0], bins=40)
            axes[0].set_title("Projected Euclidean (km)")
            comparison["haversine_distance_km"].hist(ax=axes[1], bins=40)
            axes[1].set_title("Haversine (km)")
            comparison["geodesic_distance_km"].hist(ax=axes[2], bins=40)
            axes[2].set_title("Geodesic (km)")
            plt.tight_layout()
            plt.show()
            """
        ),
        md("""
        **Interpretation**
        - Geodesic is the best reference for earth distance.
        - Haversine is usually a strong computational compromise.
        """),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_04() -> nbf.NotebookNode:
    cells = [
        md("""
        # 04 - Clustering Algorithms and Choosing K

        Compare six clustering methods and evaluate them with internal quality metrics.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.features import build_clustering_features, select_features
            from src.outlier_detection import detect_outliers
            from src.clustering import run_all
            from src.evaluation import evaluate_all, k_selection_diagnostics
            from src.visualization import plot_k_selection_metrics, plot_elbow_curve

            df = load_and_clean_data(TRAIN_FILE_PATH)
            feat = build_clustering_features(df.copy())
            X = select_features(feat)
            geo = feat[["Restaurant_latitude", "Restaurant_longitude"]].to_numpy(dtype=float)

            consensus, _ = detect_outliers(feat, feature_matrix=X)
            keep = ~consensus.mask

            feat = feat.loc[keep].reset_index(drop=True)
            X = X[keep]
            geo = geo[keep]

            results = run_all(X, geo_coords=geo)
            eval_df = evaluate_all(X, results, geo_coords=geo)
            display(eval_df)

            diag_df = k_selection_diagnostics(X)
            display(diag_df.head())
            print(plot_k_selection_metrics(diag_df, filename="nb_k_selection_metrics.png"))
            elbow_df = diag_df.rename(columns={"k":"n_clusters"})[["n_clusters","inertia","silhouette"]]
            print(plot_elbow_curve(elbow_df, filename="nb_elbow.png"))
            """
        ),
        md("""
        **Interpretation**
        - Do not pick models from one metric only.
        - Evaluate separation, compactness, stability, and noise behavior together.
        """),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_05() -> nbf.NotebookNode:
    cells = [
        md("""
        # 05 - Outliers, Business Zones, and Interpretability

        Convert technical clusters into operations-friendly business zones.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.features import build_clustering_features, select_features
            from src.outlier_detection import detect_outliers
            from src.clustering import run_algorithm
            from src.business_zones import build_cluster_kpis, assign_business_zone_labels, attach_zone_labels

            df = load_and_clean_data(TRAIN_FILE_PATH)
            feat = build_clustering_features(df.copy())
            X = select_features(feat)
            geo = feat[["Restaurant_latitude", "Restaurant_longitude"]].to_numpy(dtype=float)

            consensus, reports = detect_outliers(feat, feature_matrix=X)
            outlier_summary = pd.DataFrame({"method": [r.name for r in reports], "outliers": [r.n_outliers for r in reports]})
            display(outlier_summary)

            mask = ~consensus.mask
            feat_clean = feat.loc[mask].reset_index(drop=True)
            X_clean = X[mask]
            geo_clean = geo[mask]

            result = run_algorithm("kmeans", X_clean, geo_coords=geo_clean)
            kpis = build_cluster_kpis(feat_clean, result.labels)
            labeled_kpis, rule_summary = assign_business_zone_labels(kpis)
            labeled_rows = attach_zone_labels(feat_clean, result.labels, labeled_kpis)

            display(labeled_kpis)
            display(labeled_rows[["cluster", "zone_label", "zone_reason"]].head())
            print(rule_summary)
            """
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_06() -> nbf.NotebookNode:
    cells = [
        md("""
        # 06 - Advanced Spatial Analytics

        Build density heatmaps, KDE hotspots, and coverage diagnostics.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.features import build_clustering_features, select_features
            from src.clustering import run_algorithm
            from src.spatial_analysis import build_grid_density, kde_hotspots, service_coverage_analysis
            from src.visualization import plot_density_heatmap

            df = load_and_clean_data(TRAIN_FILE_PATH)
            feat = build_clustering_features(df.copy())
            X = select_features(feat)
            geo = feat[["Restaurant_latitude", "Restaurant_longitude"]].to_numpy(dtype=float)

            result = run_algorithm("kmeans", X, geo_coords=geo)
            grid = build_grid_density(feat)
            hotspots = kde_hotspots(feat, top_n=100)
            coverage = service_coverage_analysis(feat, result.labels)

            display(grid.head(20))
            display(hotspots.head(20))
            display(coverage)
            print(plot_density_heatmap(feat, filename="nb_density_heatmap.png"))
            """
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_07() -> nbf.NotebookNode:
    cells = [
        md("""
        # 07 - Downstream AutoML Benchmark

        Benchmark manual ML vs LazyPredict vs PyCaret vs FLAML for delivery-time prediction.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.features import build_downstream_features
            from src.downstream_benchmark import run_full_benchmark

            df = load_and_clean_data(TRAIN_FILE_PATH)
            feat = build_downstream_features(df.copy())
            bench = run_full_benchmark(feat)
            display(bench.sort_values(["status", "rmse"]))
            """
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def notebook_08() -> nbf.NotebookNode:
    cells = [
        md("""
        # 08 - End-to-End Pipeline and Streamlit Demo

        Execute the production pipeline and inspect generated reports/artifacts.
        """),
        *educational_block(),
        *common_setup_cells(),
        code(
            """
            from src.pipeline import GeospatialClusteringPipeline

            pipe = GeospatialClusteringPipeline(run_downstream_automl=True)
            report = pipe.run()

            print("Best algorithm:", report.best_algorithm)
            print("Warnings:", report.warnings)
            print("Business impact:")
            print(report.business_impact)
            print("Report files:")
            for k, v in report.report_paths.items():
                print(f"- {k}: {v}")
            """
        ),
        md("""
        ## Streamlit Demo

        Launch locally:
        ```bash
        uv run streamlit run streamlit_app/app.py
        ```
        """),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def main() -> None:
    notebooks_dir = Path(__file__).resolve().parent
    outputs = {
        "01_geospatial_foundations.ipynb": notebook_01(),
        "02_dataset_profile_and_eda.ipynb": notebook_02(),
        "03_distance_engineering.ipynb": notebook_03(),
        "04_clustering_algorithms_and_k_selection.ipynb": notebook_04(),
        "05_outliers_business_zones_and_interpretability.ipynb": notebook_05(),
        "06_advanced_spatial_analysis.ipynb": notebook_06(),
        "07_downstream_automl_benchmark.ipynb": notebook_07(),
        "08_end_to_end_pipeline_and_streamlit_demo.ipynb": notebook_08(),
    }

    for filename, notebook in outputs.items():
        path = notebooks_dir / filename
        nbf.write(notebook, path)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
