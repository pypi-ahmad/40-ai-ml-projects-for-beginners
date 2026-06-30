"""Borrower segmentation algorithms, validation metrics, and business-friendly segment naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from .config import FIGURES_DIR, RANDOM_STATE


@dataclass(slots=True)
class SegmentationOutputs:
    """Segmentation outputs for downstream reporting."""

    labels_by_algorithm: dict[str, np.ndarray]
    metrics_table: pd.DataFrame
    profile_table: pd.DataFrame
    named_segments: dict[int, str]
    best_algorithm: str


class BorrowerSegmenter:
    """Compare clustering techniques and derive interpretable borrower segments."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.scaler = RobustScaler()

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> SegmentationOutputs:
        """Run all required clustering methods and select best candidate by silhouette score."""
        output_dir = output_dir or FIGURES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = feature_cols or self._default_segmentation_features(df)
        x = df[feature_cols].copy()
        x_scaled = self.scaler.fit_transform(x)

        labels_by_algorithm: dict[str, np.ndarray] = {}
        metric_rows: list[dict[str, float | str | int]] = []

        search_spaces = self._search_spaces()
        for algorithm, configs in search_spaces.items():
            best_for_algo: dict[str, Any] | None = None
            best_score = -np.inf
            for params in configs:
                labels = self._fit_predict(algorithm, x_scaled, params)
                metrics = self._cluster_metrics(x_scaled, labels)
                if np.isnan(metrics["silhouette_score"]):
                    score = -np.inf
                else:
                    score = float(metrics["silhouette_score"])

                if score > best_score:
                    best_score = score
                    best_for_algo = {
                        "params": params,
                        "labels": labels,
                        "metrics": metrics,
                    }
            if best_for_algo is None:
                continue

            labels_by_algorithm[algorithm] = best_for_algo["labels"]
            stability = self._stability_score(algorithm, x_scaled, best_for_algo["labels"], best_for_algo["params"])
            metric_rows.append(
                {
                    "algorithm": algorithm,
                    "params": str(best_for_algo["params"]),
                    **best_for_algo["metrics"],
                    "stability_score": round(float(stability), 4),
                }
            )

        metrics_df = pd.DataFrame(metric_rows).sort_values(
            by=["silhouette_score", "stability_score", "calinski_harabasz"],
            ascending=[False, False, False],
        )
        best_algorithm = str(metrics_df.iloc[0]["algorithm"])

        best_labels = labels_by_algorithm[best_algorithm]
        profile_df = self._build_profiles(df, best_labels, feature_cols)
        segment_names = self._auto_name_segments(profile_df)

        self._plot_metrics(metrics_df, output_dir / "segmentation_metrics.png")
        self._plot_clusters_pca(x_scaled, labels_by_algorithm, output_dir / "segmentation_pca.png")

        return SegmentationOutputs(
            labels_by_algorithm=labels_by_algorithm,
            metrics_table=metrics_df.reset_index(drop=True),
            profile_table=profile_df,
            named_segments=segment_names,
            best_algorithm=best_algorithm,
        )

    @staticmethod
    def _default_segmentation_features(df: pd.DataFrame) -> list[str]:
        preferred = [
            "Monthly_Income",
            "Loan_to_Income_Ratio",
            "Debt_Burden_Score",
            "Collateral_Coverage_Ratio",
            "Missed_Payment_Severity",
            "Delinquency_Score",
            "Recovery_Difficulty_Index",
            "Collection_Intensity_Score",
            "Behavioral_Risk_Score",
        ]
        return [c for c in preferred if c in df.columns]

    @staticmethod
    def _search_spaces() -> dict[str, list[dict[str, Any]]]:
        return {
            "KMeans": [{"n_clusters": k, "n_init": 30} for k in [3, 4, 5, 6]],
            "MiniBatchKMeans": [{"n_clusters": k, "batch_size": 64} for k in [3, 4, 5, 6]],
            "Hierarchical": [{"n_clusters": k, "linkage": "ward"} for k in [3, 4, 5, 6]],
            "GMM": [
                {"n_components": k, "covariance_type": cov}
                for k in [3, 4, 5, 6]
                for cov in ["full", "diag"]
            ],
            "DBSCAN": [{"eps": eps, "min_samples": ms} for eps in [0.8, 1.0, 1.2, 1.4] for ms in [5, 8, 10]],
        }

    def _fit_predict(self, algorithm: str, x_scaled: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        if algorithm == "KMeans":
            model = KMeans(random_state=self.random_state, **params)
            return model.fit_predict(x_scaled)
        if algorithm == "MiniBatchKMeans":
            model = MiniBatchKMeans(random_state=self.random_state, **params)
            return model.fit_predict(x_scaled)
        if algorithm == "Hierarchical":
            model = AgglomerativeClustering(**params)
            return model.fit_predict(x_scaled)
        if algorithm == "GMM":
            model = GaussianMixture(random_state=self.random_state, **params)
            return model.fit_predict(x_scaled)
        if algorithm == "DBSCAN":
            model = DBSCAN(**params)
            return model.fit_predict(x_scaled)
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    @staticmethod
    def _cluster_metrics(x_scaled: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
        # DBSCAN may produce noise label -1. Metrics need at least 2 non-noise clusters.
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_x = x_scaled[valid_mask]

        if len(np.unique(valid_labels)) < 2 or len(valid_labels) < 10:
            return {
                "n_clusters": int(len(np.unique(valid_labels))),
                "silhouette_score": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "noise_ratio": round(float((labels == -1).mean()), 4),
            }

        return {
            "n_clusters": int(len(np.unique(valid_labels))),
            "silhouette_score": round(float(silhouette_score(valid_x, valid_labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(valid_x, valid_labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(valid_x, valid_labels)), 2),
            "noise_ratio": round(float((labels == -1).mean()), 4),
        }

    def _stability_score(
        self,
        algorithm: str,
        x_scaled: np.ndarray,
        base_labels: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        """Estimate segmentation stability via average adjusted rand score across seeds."""
        if algorithm in {"Hierarchical", "DBSCAN"}:
            return 1.0

        seeds = [11, 23, 37]
        scores: list[float] = []
        for seed in seeds:
            if algorithm == "KMeans":
                labels = KMeans(random_state=seed, **params).fit_predict(x_scaled)
            elif algorithm == "MiniBatchKMeans":
                labels = MiniBatchKMeans(random_state=seed, **params).fit_predict(x_scaled)
            elif algorithm == "GMM":
                labels = GaussianMixture(random_state=seed, **params).fit_predict(x_scaled)
            else:
                continue
            scores.append(adjusted_rand_score(base_labels, labels))
        return float(np.mean(scores)) if scores else np.nan

    @staticmethod
    def _build_profiles(df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
        profile_df = df.copy()
        profile_df["segment"] = labels

        grouped = profile_df.groupby("segment", as_index=False)[feature_cols].mean().sort_values("segment")
        sizes = profile_df["segment"].value_counts().rename_axis("segment").reset_index(name="segment_size")
        grouped = grouped.merge(sizes, on="segment", how="left")
        return grouped

    @staticmethod
    def _auto_name_segments(profile_df: pd.DataFrame) -> dict[int, str]:
        """Assign business-friendly segment names from profile characteristics."""
        names: dict[int, str] = {}

        income_q75 = profile_df["Monthly_Income"].quantile(0.75) if "Monthly_Income" in profile_df.columns else 0.0
        delinquency_q75 = profile_df["Delinquency_Score"].quantile(0.75) if "Delinquency_Score" in profile_df.columns else 0.0
        difficulty_q75 = (
            profile_df["Recovery_Difficulty_Index"].quantile(0.75) if "Recovery_Difficulty_Index" in profile_df.columns else 0.0
        )
        collateral_q25 = (
            profile_df["Collateral_Coverage_Ratio"].quantile(0.25) if "Collateral_Coverage_Ratio" in profile_df.columns else 0.0
        )
        burden_q75 = profile_df["Loan_to_Income_Ratio"].quantile(0.75) if "Loan_to_Income_Ratio" in profile_df.columns else 0.0

        used_names: dict[str, int] = {}
        for row in profile_df.itertuples(index=False):
            segment = int(getattr(row, "segment"))
            if segment == -1:
                name = "Anomalous Borrowers (Manual Review Queue)"
            elif (
                getattr(row, "Delinquency_Score", 0.0) >= delinquency_q75
                and getattr(row, "Recovery_Difficulty_Index", 0.0) >= difficulty_q75
                and getattr(row, "Collateral_Coverage_Ratio", 0.0) <= collateral_q25
            ):
                name = "Legal Escalation Candidate Segment"
            elif getattr(row, "Loan_to_Income_Ratio", 0.0) >= burden_q75:
                name = "High Loan Burden Segment"
            elif (
                getattr(row, "Monthly_Income", 0.0) >= income_q75
                and getattr(row, "Delinquency_Score", 0.0) < delinquency_q75
                and getattr(row, "Recovery_Difficulty_Index", 0.0) < difficulty_q75
            ):
                name = "High Income Low Risk Segment"
            elif (
                getattr(row, "Collateral_Coverage_Ratio", 0.0) > max(1.0, collateral_q25)
                and getattr(row, "Recovery_Difficulty_Index", 0.0) < difficulty_q75
            ):
                name = "Recovery Friendly Segment"
            else:
                name = "Moderate Income Stable Borrowers Segment"

            # Ensure unique labels while keeping business language.
            used_names[name] = used_names.get(name, 0) + 1
            if used_names[name] > 1:
                name = f"{name} ({used_names[name]})"
            names[segment] = name
        return names

    @staticmethod
    def _plot_metrics(metrics_df: pd.DataFrame, path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.barplot(data=metrics_df, x="algorithm", y="silhouette_score", hue="algorithm", legend=False, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Silhouette Score (Higher Better)")
        axes[0].tick_params(axis="x", rotation=30)

        sns.barplot(data=metrics_df, x="algorithm", y="davies_bouldin", hue="algorithm", legend=False, ax=axes[1], palette="Oranges")
        axes[1].set_title("Davies-Bouldin Index (Lower Better)")
        axes[1].tick_params(axis="x", rotation=30)

        sns.barplot(data=metrics_df, x="algorithm", y="calinski_harabasz", hue="algorithm", legend=False, ax=axes[2], palette="Greens")
        axes[2].set_title("Calinski-Harabasz (Higher Better)")
        axes[2].tick_params(axis="x", rotation=30)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _plot_clusters_pca(x_scaled: np.ndarray, labels_by_algorithm: dict[str, np.ndarray], path: Path) -> None:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        projected = pca.fit_transform(x_scaled)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for idx, (name, labels) in enumerate(labels_by_algorithm.items()):
            ax = axes[idx]
            scatter = ax.scatter(projected[:, 0], projected[:, 1], c=labels, cmap="Spectral", s=18, alpha=0.8)
            ax.set_title(name)
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
            fig.colorbar(scatter, ax=ax)

        for idx in range(len(labels_by_algorithm), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

