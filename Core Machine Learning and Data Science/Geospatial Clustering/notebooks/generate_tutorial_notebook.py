"""Generate tutorial notebook for geospatial clustering project."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Geospatial Clustering for Food Delivery Optimization

            **Zero-to-hero tutorial notebook** for delivery analytics with geospatial ML.

            You will learn how to:
            1. Load and clean noisy logistics data.
            2. Engineer geospatial + temporal + operational features.
            3. Detect outliers with multi-method consensus.
            4. Cluster delivery patterns using six algorithms.
            5. Evaluate cluster quality and convert outputs to business insights.

            Dataset: `data/raw/train.csv` from Kaggle (`gauravmalik26/food-delivery-dataset`).
            """
        ),
        md(
            """
            ## 1) Why Geospatial Clustering Matters

            Food delivery operations generate high-volume location traces. Clustering helps answer:

            - Which zones produce dense order demand?
            - Where do delivery routes become slow due to traffic and distance?
            - How can we design service zones for better ETA and lower delivery cost?

            This notebook focuses on **unsupervised pattern discovery**.
            """
        ),
        code(
            """
            # 1. Imports and environment setup
            from __future__ import annotations

            import sys
            from pathlib import Path
            import warnings

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            warnings.filterwarnings("ignore")

            # Make notebook runnable from both project root and notebooks/ folder
            if (Path.cwd() / "src").exists():
                PROJECT_ROOT = Path.cwd()
            elif (Path.cwd().parent / "src").exists():
                PROJECT_ROOT = Path.cwd().parent
            else:
                raise RuntimeError("Could not find project root with src/ directory.")

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            print(f"Project root: {PROJECT_ROOT}")
            """
        ),
        code(
            """
            # 2. Project module imports
            from src.data_loader import load_and_clean_data
            from src.features import build_features, select_features
            from src.clustering import run_all
            from src.evaluation import evaluate_all, elbow_curve
            from src.outlier_detection import detect_outliers
            from src.visualization import (
                plot_clusters_2d,
                plot_delivery_map,
                plot_elbow_curve,
                plot_silhouette,
                plot_feature_distributions,
                plot_interactive_map,
            )
            from src.pipeline import GeospatialClusteringPipeline
            """
        ),
        md(
            """
            ## 2) Load Raw Data

            We start with raw CSV inspection before any transformations.
            """
        ),
        code(
            """
            RAW_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
            raw_df = pd.read_csv(RAW_PATH)

            print("Raw shape:", raw_df.shape)
            display(raw_df.head(3))
            display(raw_df.describe(include="all").T.head(10))
            """
        ),
        md(
            """
            ## 3) Data Cleaning and Validation

            `load_and_clean_data` applies:

            - whitespace normalization
            - date/time parsing
            - numeric casting
            - weather/time text cleaning
            - coordinate sanity filtering (India bounds)
            """
        ),
        code(
            """
            df = load_and_clean_data(str(RAW_PATH), validate=True)
            report = df.attrs.get("validation_report", {})

            print("Clean shape:", df.shape)
            print("Validation report:")
            print(report)

            display(df.head(3))
            """
        ),
        code(
            """
            # Quick data quality audit
            null_share = (df.isna().mean() * 100).sort_values(ascending=False)
            display(null_share.head(15).to_frame("missing_pct"))

            city_dist = df["City"].value_counts(dropna=False)
            display(city_dist.to_frame("count"))
            """
        ),
        md(
            """
            ## 4) Feature Engineering

            Clustering quality depends heavily on feature design.

            Engineered features include:
            - delivery distance (haversine)
            - temporal dimensions (hour/day/month/day-of-week)
            - speed proxy (`distance / duration`)
            - encoded traffic, city, festival indicators
            - operational columns like vehicle condition and multi-deliveries
            """
        ),
        code(
            """
            feat_df = build_features(df.copy())
            X = select_features(feat_df)

            print("Feature dataframe shape:", feat_df.shape)
            print("Clustering matrix shape:", X.shape)
            display(feat_df.head(3))
            """
        ),
        code(
            """
            # Feature snapshots for intuition
            interesting_cols = [
                "delivery_distance_km",
                "duration_min",
                "speed_kmph",
                "traffic_code",
                "city_code",
                "festival_binary",
            ]

            display(feat_df[interesting_cols].describe().T)
            """
        ),
        md(
            """
            ## 5) Outlier Detection Before Clustering

            Outliers distort centroid-based methods. We aggregate signals from:

            - coordinate bounds
            - IQR thresholds
            - Mahalanobis distance
            - Isolation Forest

            Consensus rule: mark point as outlier if >= 2 methods agree.
            """
        ),
        code(
            """
            consensus_report, individual_reports = detect_outliers(
                feat_df,
                feature_matrix=X,
                methods=None,
                min_consensus_votes=2,
            )

            print("Consensus outliers:", consensus_report.n_outliers)
            for r in individual_reports:
                print(f"- {r.name}: {r.n_outliers}")

            clean_mask = ~consensus_report.mask
            feat_clean = feat_df.loc[clean_mask].reset_index(drop=True)
            X_clean = X[clean_mask]

            print("Rows after outlier removal:", X_clean.shape[0])
            """
        ),
        md(
            """
            ## 6) Clustering Algorithms

            We compare six methods:

            1. K-Means
            2. MiniBatch K-Means
            3. DBSCAN
            4. HDBSCAN
            5. Agglomerative Clustering
            6. Gaussian Mixture Model

            To keep runtime practical in notebook environments, we use optional sampling.
            """
        ),
        code(
            """
            FAST_MODE = True
            SAMPLE_SIZE = 12000
            rng = np.random.default_rng(42)

            if FAST_MODE and len(X_clean) > SAMPLE_SIZE:
                sample_idx = rng.choice(len(X_clean), size=SAMPLE_SIZE, replace=False)
                sample_idx = np.sort(sample_idx)
                X_work = X_clean[sample_idx]
                feat_work = feat_clean.iloc[sample_idx].reset_index(drop=True)
            else:
                X_work = X_clean
                feat_work = feat_clean.copy()

            print("Working matrix shape:", X_work.shape)
            """
        ),
        code(
            """
            clustering_results = run_all(X_work)
            eval_df = evaluate_all(X_work, clustering_results)

            display(eval_df)
            """
        ),
        code(
            """
            best_algo = eval_df.iloc[0]["algorithm"]
            best_labels = clustering_results[best_algo].labels

            print("Best algorithm by silhouette:", best_algo)
            print("Best silhouette:", eval_df.iloc[0]["silhouette"])
            """
        ),
        md(
            """
            ## 7) Visual Diagnostics

            We generate core diagnostics:

            - Elbow curve (`k` selection intuition)
            - 2D cluster projection
            - Geographic cluster map
            - Silhouette shape
            - Feature distributions by cluster
            - Interactive Folium map
            """
        ),
        code(
            """
            elbow_df = elbow_curve(X_work, k_range=range(2, 10))
            elbow_path = plot_elbow_curve(elbow_df, filename="nb_elbow_curve.png")

            # Use first two dimensions for quick 2D projection
            cluster_plot_path = plot_clusters_2d(
                X_work[:, :2],
                best_labels,
                title=f"Notebook Clusters ({best_algo})",
                filename="nb_clusters_2d.png",
            )

            delivery_map_path = plot_delivery_map(
                feat_work,
                best_labels,
                title=f"Delivery Map ({best_algo})",
                filename="nb_delivery_map.png",
            )

            sil_path = plot_silhouette(
                X_work,
                best_labels,
                filename="nb_silhouette.png",
            )

            dist_path = plot_feature_distributions(
                feat_work,
                best_labels,
                features=["delivery_distance_km", "speed_kmph"],
                filename="nb_feature_distributions.png",
            )

            print("Saved plots:")
            for p in [elbow_path, cluster_plot_path, delivery_map_path, sil_path, dist_path]:
                print("-", p)
            """
        ),
        code(
            """
            # Interactive map export
            interactive_map_path = plot_interactive_map(
                feat_work,
                best_labels,
                title=f"Interactive Delivery Clusters ({best_algo})",
                filename="nb_interactive_map.html",
            )

            print("Interactive map saved to:", interactive_map_path)
            """
        ),
        md(
            """
            ## 8) Business Insight Layer

            Raw cluster IDs are not business value until translated.
            We aggregate key operational signals by cluster.
            """
        ),
        code(
            """
            analysis = feat_work.copy()
            analysis["cluster"] = best_labels

            cluster_kpis = (
                analysis.groupby("cluster", as_index=False)
                .agg(
                    orders=("cluster", "count"),
                    avg_distance_km=("delivery_distance_km", "mean"),
                    avg_duration_min=("duration_min", "mean"),
                    avg_speed_kmph=("speed_kmph", "mean"),
                    avg_traffic_code=("traffic_code", "mean"),
                    avg_vehicle_condition=("Vehicle_condition", "mean"),
                )
                .sort_values("orders", ascending=False)
            )

            display(cluster_kpis)
            """
        ),
        md(
            """
            ### Interpreting clusters for operations

            Use this decision template:

            - **High orders + low distance + high speed**: candidate for SLA tightening and batching.
            - **High orders + high traffic + low speed**: candidate for rider rebalancing and surge planning.
            - **Low orders + long distance**: candidate for dynamic pricing or service-radius redesign.

            Extend this notebook by joining financial metrics (cost per km, cancellation, tip rates) for full unit economics.
            """
        ),
        md(
            """
            ## 9) One-call Pipeline Demo

            If you want single-command execution with reports and plots, use pipeline wrapper.
            """
        ),
        code(
            """
            demo_pipeline = GeospatialClusteringPipeline(
                data_path=str(RAW_PATH),
                algorithms=["kmeans", "minibatch_kmeans", "dbscan"],
                remove_outliers=True,
            )

            demo_report = demo_pipeline.run()
            print("Pipeline best algorithm:", demo_report.best_algorithm)
            print("Pipeline output dir:", demo_report.output_dir)
            display(pd.DataFrame(demo_report.algorithm_results).T)
            """
        ),
        md(
            """
            ## 10) Next Steps

            1. Tune algorithm-specific hyperparameters by city segment.
            2. Add demand forecasting per cluster.
            3. Add route-time simulation and dispatch optimization.
            4. Deploy real-time scoring with geofencing.

            Notebook complete.
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    return nb


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "notebooks" / "01_geospatial_clustering_tutorial.ipynb"

    notebook = build_notebook()
    with out_path.open("w", encoding="utf-8") as f:
        nbf.write(notebook, f)

    print(f"Notebook written: {out_path}")


if __name__ == "__main__":
    main()
