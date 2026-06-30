"""Exploratory data analysis utilities for loan recovery dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TARGET_COLUMN

sns.set_theme(style="whitegrid", context="talk")


@dataclass(slots=True)
class EDAOutputs:
    """Container of generated EDA artifact paths."""

    target_distribution: Path
    missing_values: Path
    correlation_heatmap: Path
    numeric_histograms: Path
    boxplots: Path
    violin_plots: Path
    relationship_grid: Path


class LoanEDA:
    """Generate descriptive statistics and educational EDA plots."""

    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> None:
        self.df = df.copy()
        self.target_col = target_col

    def summary_table(self) -> pd.DataFrame:
        """Return dataset summary with null percentages and cardinality."""
        rows: list[dict[str, float | int | str]] = []
        for col in self.df.columns:
            series = self.df[col]
            rows.append(
                {
                    "column": col,
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_pct": round(float(series.isna().mean() * 100), 2),
                    "unique_values": int(series.nunique(dropna=True)),
                    "sample_min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                    "sample_max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def correlation_matrix(self) -> pd.DataFrame:
        """Compute numeric correlation matrix."""
        numeric = self.df.select_dtypes(include=[np.number])
        return numeric.corr(numeric_only=True)

    def relationship_analysis(self) -> pd.DataFrame:
        """Compute recovery rate by key business dimensions."""
        df = self.df.copy()
        df["Recovery_Rate_Score"] = df[self.target_col].map({0: 1.0, 1: 0.6, 2: 0.1})
        group_cols = [
            "Employment_Status",
            "Loan_Purpose",
            "Residence_Type",
        ]
        rows = []
        for col in group_cols:
            grouped = (
                df.groupby(col, as_index=False)
                .agg(
                    avg_recovery_score=("Recovery_Rate_Score", "mean"),
                    avg_days_past_due=("Days_Past_Due", "mean"),
                    avg_outstanding=("Outstanding_Balance", "mean"),
                    borrowers=("Borrower_ID", "count"),
                )
                .sort_values("avg_recovery_score", ascending=False)
            )
            grouped["dimension"] = col
            grouped = grouped.rename(columns={col: "segment"})
            rows.append(grouped)
        return pd.concat(rows, ignore_index=True)

    def generate_all_plots(self, output_dir: Path | None = None) -> EDAOutputs:
        """Generate required EDA visuals and return artifact locations."""
        output_dir = output_dir or FIGURES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        target_path = self._plot_target_distribution(output_dir)
        missing_path = self._plot_missing_values(output_dir)
        corr_path = self._plot_correlation_heatmap(output_dir)
        hist_path = self._plot_histograms(output_dir)
        box_path = self._plot_boxplots(output_dir)
        violin_path = self._plot_violin_plots(output_dir)
        rel_path = self._plot_relationship_grid(output_dir)

        return EDAOutputs(
            target_distribution=target_path,
            missing_values=missing_path,
            correlation_heatmap=corr_path,
            numeric_histograms=hist_path,
            boxplots=box_path,
            violin_plots=violin_path,
            relationship_grid=rel_path,
        )

    def _plot_target_distribution(self, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        counts = self.df[self.target_col].value_counts().sort_index()
        labels = ["Fully Recovered", "Partially Recovered", "Written Off"]
        plot_df = pd.DataFrame({"label": labels[: len(counts)], "count": counts.values})
        sns.barplot(data=plot_df, x="label", y="count", hue="label", legend=False, palette="viridis", ax=ax)
        ax.set_title("Recovery Status Distribution")
        ax.set_xlabel("Recovery Status")
        ax.set_ylabel("Borrower Count")
        for idx, value in enumerate(counts.values):
            ax.text(idx, value + 2, str(value), ha="center", va="bottom", fontsize=10)
        path = output_dir / "eda_target_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_missing_values(self, output_dir: Path) -> Path:
        missing = self.df.isna().sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df = pd.DataFrame({"column": missing.index, "count": missing.values})
        sns.barplot(data=plot_df, x="column", y="count", hue="column", legend=False, ax=ax, palette="mako")
        ax.set_title("Missing Values by Column")
        ax.tick_params(axis="x", rotation=75)
        path = output_dir / "eda_missing_values.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_correlation_heatmap(self, output_dir: Path) -> Path:
        corr = self.correlation_matrix()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Numeric Correlation Matrix")
        path = output_dir / "eda_correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_histograms(self, output_dir: Path) -> Path:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col][:9]
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(self.df, x=col, hue=self.target_col, kde=True, ax=axes[i], palette="Set2")
            axes[i].set_title(f"{col} Distribution")
        for j in range(len(numeric_cols), len(axes)):
            axes[j].set_visible(False)
        path = output_dir / "eda_numeric_histograms.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_boxplots(self, output_dir: Path) -> Path:
        key_cols = [
            "Monthly_Income",
            "Loan_Amount",
            "Estimated_EMI" if "Estimated_EMI" in self.df.columns else "Debt_to_Income_Ratio",
            "Missed_Payments",
            "Days_Past_Due",
            "Collateral_Value",
        ]
        key_cols = [c for c in key_cols if c in self.df.columns]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, col in enumerate(key_cols):
            sns.boxplot(data=self.df, x=self.target_col, y=col, hue=self.target_col, ax=axes[i], legend=False, palette="pastel")
            axes[i].set_title(f"{col} vs Recovery Status")
        for j in range(len(key_cols), len(axes)):
            axes[j].set_visible(False)
        path = output_dir / "eda_boxplots.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_violin_plots(self, output_dir: Path) -> Path:
        key_cols = [
            "Monthly_Income",
            "Loan_Amount",
            "Missed_Payments",
            "Days_Past_Due",
            "Collateral_Value",
            "Debt_to_Income_Ratio",
        ]
        key_cols = [c for c in key_cols if c in self.df.columns]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, col in enumerate(key_cols):
            sns.violinplot(
                data=self.df,
                x=self.target_col,
                y=col,
                hue=self.target_col,
                ax=axes[i],
                palette="Set3",
                inner="quartile",
                legend=False,
            )
            axes[i].set_title(f"{col} by Recovery Status")
        for j in range(len(key_cols), len(axes)):
            axes[j].set_visible(False)
        path = output_dir / "eda_violin_plots.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_relationship_grid(self, output_dir: Path) -> Path:
        pairs = [
            ("Monthly_Income", "Recovery_Status"),
            ("Loan_Amount", "Recovery_Status"),
            ("Estimated_EMI" if "Estimated_EMI" in self.df.columns else "Debt_to_Income_Ratio", "Recovery_Status"),
            ("Missed_Payments", "Recovery_Status"),
            ("Days_Past_Due", "Recovery_Status"),
            ("Collateral_Value", "Recovery_Status"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for idx, (x_col, y_col) in enumerate(pairs):
            if x_col not in self.df.columns:
                axes[idx].set_visible(False)
                continue
            sns.kdeplot(
                data=self.df,
                x=x_col,
                hue=y_col,
                fill=True,
                common_norm=False,
                alpha=0.3,
                ax=axes[idx],
            )
            axes[idx].set_title(f"KDE: {x_col} by Recovery Status")
        path = output_dir / "eda_relationship_grid.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
