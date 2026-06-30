"""End-to-end geospatial clustering pipeline.

The pipeline is production-oriented and reusable:
- dataset load/clean/profile
- geospatial math + CRS validation
- feature engineering and distance modeling
- outlier detection
- multi-algorithm clustering and evaluation
- business-zone labeling
- spatial analytics + map artifacts
- optional downstream AutoML benchmark
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.business_zones import assign_business_zone_labels, attach_zone_labels, build_cluster_kpis
from src.clustering import ClusteringResult, run_algorithm
from src.config import (
    ASSIGNMENTS_PATH,
    BENCHMARK_REPORT_PATH,
    CLUSTER_EVAL_REPORT_PATH,
    CLUSTERING_ALGORITHMS,
    COL_OUTLIER,
    COL_OUTLIER_SCORE,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    GEOSPATIAL_VALIDATION_REPORT_PATH,
    OUTPUT_DIR,
    PIPELINE_REPORT_PATH,
    PROFILE_REPORT_PATH,
    SPATIAL_SUMMARY_REPORT_PATH,
    TRAIN_FILE_PATH,
    ZONE_KPI_REPORT_PATH,
)
from src.data_loader import load_and_clean_data, load_raw_data
from src.data_quality import run_quality_gate, save_quality_report
from src.downstream_benchmark import run_full_benchmark
from src.evaluation import (
    compute_business_impact_metrics,
    evaluate_all,
    evaluate_metric_table,
    k_selection_diagnostics,
)
from src.features import build_clustering_features, build_downstream_features, select_features
from src.geospatial_validation import (
    build_geospatial_validation_report,
    save_geospatial_validation_report,
)
from src.outlier_detection import OutlierReport, detect_outliers
from src.spatial_analysis import build_grid_density, kde_hotspots, service_coverage_analysis
from src.visualization import (
    close_all,
    plot_cluster_boundaries,
    plot_clusters_2d,
    plot_delivery_map,
    plot_density_heatmap,
    plot_elbow_curve,
    plot_feature_distributions,
    plot_interactive_map,
    plot_k_selection_metrics,
    plot_outlier_map,
    plot_silhouette,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Structured summary of one pipeline run."""

    n_raw_rows: int
    n_clean_rows: int
    n_samples: int
    n_features: int
    n_outliers_removed: int
    algorithm_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_algorithms: list[str] = field(default_factory=list)
    best_algorithm: str | None = None
    business_impact: dict[str, float] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)
    geospatial_validation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    output_dir: str = ""


class GeospatialClusteringPipeline:
    """Reusable geospatial clustering pipeline."""

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        remove_outliers: bool = True,
        outlier_methods: list[str] | None = None,
        algorithms: list[str] | None = None,
        run_downstream_automl: bool = False,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else TRAIN_FILE_PATH
        self.remove_outliers = remove_outliers
        self.outlier_methods = outlier_methods
        self.algorithms = algorithms or list(CLUSTERING_ALGORITHMS)
        self.run_downstream_automl = run_downstream_automl

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None
        self.downstream_df: pd.DataFrame | None = None
        self.x_matrix: np.ndarray | None = None
        self.geo_coords: np.ndarray | None = None
        self.clean_mask: np.ndarray | None = None
        self.outlier_consensus: OutlierReport | None = None
        self.outlier_reports: list[OutlierReport] = []
        self.original_outlier_mask: np.ndarray | None = None
        self.results: dict[str, ClusteringResult] = {}
        self.metrics_df: pd.DataFrame = pd.DataFrame()
        self.assignments_df: pd.DataFrame | None = None

        self.report = PipelineReport(
            n_raw_rows=0,
            n_clean_rows=0,
            n_samples=0,
            n_features=0,
            n_outliers_removed=0,
        )

    def run(self) -> PipelineReport:
        """Execute the full pipeline and write artifacts to ``outputs``."""
        logger.info("Pipeline started: %s", self.data_path)
        close_all()

        try:
            self._load_and_profile_data()
            self._run_geospatial_validations()
            self._engineer_features()
            self._detect_outliers()
            self._run_clustering()
            self._evaluate()
            self._build_business_zones()
            self._run_spatial_analysis()
            self._generate_visualizations()

            if self.run_downstream_automl:
                self._run_downstream_benchmark()
        finally:
            # Save partial report even if a downstream step fails.
            self._save_report()

        logger.info("Pipeline complete. best_algorithm=%s", self.report.best_algorithm)
        return self.report

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _load_and_profile_data(self) -> None:
        self.raw_df = load_raw_data(self.data_path)
        quality = run_quality_gate(self.raw_df)
        quality_path = save_quality_report(quality, path=PROFILE_REPORT_PATH)
        self.report.report_paths["dataset_profile"] = str(quality_path)

        self.df = load_and_clean_data(self.data_path, validate=True)

        self.report.n_raw_rows = len(self.raw_df)
        self.report.n_clean_rows = len(self.df)

    def _run_geospatial_validations(self) -> None:
        validation = build_geospatial_validation_report()
        save_geospatial_validation_report(validation, GEOSPATIAL_VALIDATION_REPORT_PATH)
        self.report.report_paths["geospatial_validation"] = str(GEOSPATIAL_VALIDATION_REPORT_PATH)
        self.report.geospatial_validation = {
            "all_distance_cases_pass": validation.all_distance_cases_pass,
            "all_crs_checks_pass": validation.all_crs_checks_pass,
        }
        if not validation.all_distance_cases_pass:
            self.report.warnings.append("Distance validation failed for one or more known coordinate cases.")
        if not validation.all_crs_checks_pass:
            self.report.warnings.append("CRS validation checks failed.")

    def _engineer_features(self) -> None:
        self.features_df = build_clustering_features(self.df.copy())
        self.downstream_df = build_downstream_features(self.df.copy())
        self.x_matrix = select_features(self.features_df)
        self.geo_coords = self.features_df[[COL_RESTAURANT_LAT, COL_RESTAURANT_LON]].to_numpy(dtype=float)

        self.report.n_samples = int(self.x_matrix.shape[0])
        self.report.n_features = int(self.x_matrix.shape[1])

    def _detect_outliers(self) -> None:
        consensus, reports = detect_outliers(
            self.features_df,
            feature_matrix=self.x_matrix,
            methods=self.outlier_methods,
            min_consensus_votes=2,
        )
        self.outlier_consensus = consensus
        self.outlier_reports = reports

        outlier_mask = consensus.mask
        self.original_outlier_mask = outlier_mask.copy()
        self.features_df[COL_OUTLIER] = outlier_mask

        vote_count = np.zeros(len(self.features_df), dtype=int)
        for report in reports:
            vote_count += report.mask.astype(int)
        self.features_df[COL_OUTLIER_SCORE] = vote_count

        removed = int(outlier_mask.sum())
        self.report.n_outliers_removed = removed

        if self.remove_outliers:
            self.clean_mask = ~outlier_mask
            self.features_df = self.features_df.loc[self.clean_mask].reset_index(drop=True)
            self.df = self.df.loc[self.clean_mask].reset_index(drop=True)
            self.downstream_df = self.downstream_df.loc[self.clean_mask].reset_index(drop=True)
            self.x_matrix = self.x_matrix[self.clean_mask]
            self.geo_coords = self.geo_coords[self.clean_mask]
            self.features_df[COL_OUTLIER] = False
        else:
            self.clean_mask = np.ones(len(self.features_df), dtype=bool)

    def _run_clustering(self) -> None:
        results: dict[str, ClusteringResult] = {}
        failed: list[str] = []

        for algorithm in self.algorithms:
            try:
                results[algorithm] = run_algorithm(
                    algorithm,
                    self.x_matrix,
                    geo_coords=self.geo_coords,
                )
            except Exception:
                logger.exception("Algorithm failed: %s", algorithm)
                failed.append(algorithm)

        self.results = results
        self.report.failed_algorithms = failed

    def _evaluate(self) -> None:
        if not self.results:
            self.metrics_df = pd.DataFrame()
            self.report.best_algorithm = None
            return

        self.metrics_df = evaluate_all(self.x_matrix, self.results, geo_coords=self.geo_coords)
        self.metrics_df.to_csv(CLUSTER_EVAL_REPORT_PATH, index=False)
        self.report.report_paths["clustering_metrics"] = str(CLUSTER_EVAL_REPORT_PATH)

        diagnostics = k_selection_diagnostics(self.x_matrix)
        diagnostics_path = CLUSTER_EVAL_REPORT_PATH.with_name("k_selection_diagnostics.csv")
        diagnostics.to_csv(diagnostics_path, index=False)
        self.report.report_paths["k_selection_diagnostics"] = str(diagnostics_path)

        if self.metrics_df.empty:
            self.report.best_algorithm = None
            return

        self.report.best_algorithm = str(self.metrics_df.iloc[0]["algorithm"])

        if bool((self.metrics_df["silhouette"] < 0).all()):
            self.report.warnings.append(
                "All evaluated algorithms produced negative silhouette scores; cluster separation is weak."
            )

        for _, row in self.metrics_df.iterrows():
            algorithm = str(row["algorithm"])
            self.report.algorithm_results[algorithm] = {
                "n_clusters": int(row["n_clusters"]),
                "n_noise": int(row["n_noise"]),
                "metrics": {
                    "silhouette": float(row["silhouette"]) if pd.notna(row["silhouette"]) else float("nan"),
                    "davies_bouldin": float(row["davies_bouldin"]) if pd.notna(row["davies_bouldin"]) else float("nan"),
                    "calinski_harabasz": float(row["calinski_harabasz"]) if pd.notna(row["calinski_harabasz"]) else float("nan"),
                    "stability_ari": float(row.get("stability_ari", float("nan"))),
                    "composite_score": float(row.get("composite_score", float("nan"))),
                },
            }

    def _build_business_zones(self) -> None:
        if not self.report.best_algorithm:
            return

        best_labels = self.results[self.report.best_algorithm].labels
        current_outlier = self.features_df[COL_OUTLIER].to_numpy(dtype=bool)

        kpi = build_cluster_kpis(self.features_df, best_labels, outlier_mask=current_outlier)
        labeled_kpi, rule_summary = assign_business_zone_labels(kpi)
        labeled_kpi.to_csv(ZONE_KPI_REPORT_PATH, index=False)

        assignments = attach_zone_labels(self.features_df, best_labels, labeled_kpi)
        assignments[COL_OUTLIER] = current_outlier
        assignments.to_csv(ASSIGNMENTS_PATH, index=False)

        self.assignments_df = assignments
        self.report.report_paths["zone_kpis"] = str(ZONE_KPI_REPORT_PATH)
        self.report.report_paths["assignments"] = str(ASSIGNMENTS_PATH)
        self.report.report_paths["zone_rule_summary"] = json.dumps(asdict(rule_summary))

        business_impact = compute_business_impact_metrics(
            assignments,
            best_labels,
            outlier_mask=current_outlier,
        )
        business_impact.update(evaluate_metric_table(self.metrics_df))

        original_total = max(self.report.n_clean_rows, 1)
        business_impact["original_outlier_pct"] = float((self.report.n_outliers_removed / original_total) * 100)

        self.report.business_impact = business_impact

    def _run_spatial_analysis(self) -> None:
        if self.assignments_df is None or self.report.best_algorithm is None:
            return

        try:
            best_labels = self.results[self.report.best_algorithm].labels
            density = build_grid_density(self.assignments_df)
            hotspots = kde_hotspots(self.assignments_df)
            coverage = service_coverage_analysis(self.assignments_df, best_labels)

            spatial_path = SPATIAL_SUMMARY_REPORT_PATH
            coverage.to_csv(spatial_path, index=False)
            hotspot_path = spatial_path.with_name("kde_hotspots.csv")
            hotspots.to_csv(hotspot_path, index=False)
            density_path = spatial_path.with_name("grid_density.csv")
            density.to_csv(density_path, index=False)

            self.report.report_paths["spatial_coverage"] = str(spatial_path)
            self.report.report_paths["kde_hotspots"] = str(hotspot_path)
            self.report.report_paths["grid_density"] = str(density_path)
        except Exception:
            logger.exception("Spatial analysis failed; continuing with remaining report.")

    def _generate_visualizations(self) -> None:
        if not self.results:
            return

        try:
            diagnostics = pd.read_csv(self.report.report_paths["k_selection_diagnostics"])
            self.report.artifact_paths["k_selection_metrics"] = plot_k_selection_metrics(
                diagnostics,
                filename="k_selection_metrics.png",
            )
            elbow_df = diagnostics.rename(columns={"k": "n_clusters"})[["n_clusters", "inertia", "silhouette"]]
            self.report.artifact_paths["elbow_curve"] = plot_elbow_curve(
                elbow_df,
                filename="elbow_curve.png",
            )

            from sklearn.decomposition import PCA

            x_2d = PCA(n_components=2, random_state=42).fit_transform(self.x_matrix)

            for algorithm, result in self.results.items():
                suffix = algorithm
                try:
                    self.report.artifact_paths[f"clusters_{algorithm}"] = plot_clusters_2d(
                        x_2d,
                        result.labels,
                        title=f"Clusters ({algorithm})",
                        filename=f"clusters_{suffix}.png",
                    )
                    self.report.artifact_paths[f"delivery_map_{algorithm}"] = plot_delivery_map(
                        self.features_df,
                        result.labels,
                        title=f"Delivery Map ({algorithm})",
                        filename=f"delivery_map_{suffix}.png",
                    )
                    self.report.artifact_paths[f"boundaries_{algorithm}"] = plot_cluster_boundaries(
                        self.features_df,
                        result.labels,
                        title=f"Cluster Boundaries ({algorithm})",
                        filename=f"boundaries_{suffix}.png",
                    )
                    self.report.artifact_paths[f"silhouette_{algorithm}"] = plot_silhouette(
                        self.x_matrix,
                        result.labels,
                        filename=f"silhouette_{suffix}.png",
                    )
                    self.report.artifact_paths[f"feature_distributions_{algorithm}"] = plot_feature_distributions(
                        self.features_df,
                        result.labels,
                        filename=f"feature_distributions_{suffix}.png",
                    )
                except Exception:
                    logger.exception("Visualization failed for algorithm=%s", algorithm)

            if self.report.best_algorithm:
                best_labels = self.results[self.report.best_algorithm].labels
                self.report.artifact_paths["interactive_map"] = plot_interactive_map(
                    self.features_df,
                    best_labels,
                    title=f"Interactive Clusters ({self.report.best_algorithm})",
                    filename="interactive_map.html",
                )
                outlier_mask = self.features_df[COL_OUTLIER].to_numpy(dtype=bool)
                self.report.artifact_paths["outlier_map"] = plot_outlier_map(
                    self.features_df,
                    outlier_mask,
                    filename="outlier_map.png",
                )
                self.report.artifact_paths["density_heatmap"] = plot_density_heatmap(
                    self.features_df,
                    filename="density_heatmap.png",
                )
        except Exception:
            logger.exception("Global visualization step failed; report will still be saved.")

    def _run_downstream_benchmark(self) -> None:
        if self.downstream_df is None:
            return

        benchmark_df = run_full_benchmark(self.downstream_df, save_path=BENCHMARK_REPORT_PATH)
        self.report.report_paths["downstream_benchmark"] = str(BENCHMARK_REPORT_PATH)
        if not benchmark_df.empty:
            best_ok = benchmark_df[benchmark_df["status"] == "ok"]
            if not best_ok.empty:
                top = best_ok.sort_values("rmse").iloc[0]
                self.report.business_impact["downstream_best_rmse"] = float(top["rmse"])
                self.report.business_impact["downstream_best_model"] = str(top["model"])

    def _save_report(self) -> None:
        self.report.output_dir = str(OUTPUT_DIR)
        PIPELINE_REPORT_PATH.write_text(
            json.dumps(asdict(self.report), indent=2, default=_json_default),
            encoding="utf-8",
        )


def _json_default(value: Any) -> Any:
    """Safely serialize numpy and path objects for report JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
