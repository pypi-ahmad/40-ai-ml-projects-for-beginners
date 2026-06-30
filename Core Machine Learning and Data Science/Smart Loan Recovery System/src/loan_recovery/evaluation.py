"""Model evaluation with statistical, calibration, and business-impact metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .config import DEFAULT_FALSE_NEGATIVE_COST, DEFAULT_FALSE_POSITIVE_COST, FIGURES_DIR


@dataclass(slots=True)
class EvaluationResults:
    """Container for key evaluation metrics."""

    classification_metrics: dict[str, float]
    business_metrics: dict[str, float]
    threshold_metrics: dict[str, float]


@dataclass(slots=True)
class ThresholdOptimizationResult:
    """Result of optimizing written-off class threshold under cost objective."""

    threshold: float
    false_positive_cost: float
    false_negative_cost: float
    total_cost: float
    recall_high_risk: float
    precision_high_risk: float


class ModelEvaluator:
    """Compute quality metrics, optimize business thresholds, and generate diagnostics."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
        portfolio_df: pd.DataFrame,
        false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
        false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
    ) -> EvaluationResults:
        """Compute ML, business, and threshold metrics in one call."""
        cls = self.classification_metrics(y_true, y_pred, y_prob)
        biz = self.business_metrics(
            y_true,
            y_pred,
            y_prob,
            portfolio_df,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        threshold_metrics = {
            "high_risk_recall": biz["high_risk_detection_rate"],
            "high_risk_precision": biz["high_risk_precision"],
            "total_misclassification_cost": biz["total_misclassification_cost"],
        }
        return EvaluationResults(
            classification_metrics=cls,
            business_metrics=biz,
            threshold_metrics=threshold_metrics,
        )

    @staticmethod
    def classification_metrics(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
    ) -> dict[str, float]:
        """Compute core classification metrics including ROC-AUC and PR-AUC."""
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)
        metrics = {
            "accuracy": round(float(accuracy_score(y_true_s, y_pred_s)), 4),
            "precision_weighted": round(float(precision_score(y_true_s, y_pred_s, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_true_s, y_pred_s, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true_s, y_pred_s, average="weighted", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)), 4),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc_ovr"] = round(float(roc_auc_score(y_true_s, y_prob, multi_class="ovr")), 4)
            except Exception:
                metrics["roc_auc_ovr"] = np.nan

            high_risk_true = (y_true_s == 2).astype(int)
            metrics["pr_auc_high_risk"] = round(float(average_precision_score(high_risk_true, y_prob[:, 2])), 4)
            try:
                metrics["brier_high_risk"] = round(float(brier_score_loss(high_risk_true, y_prob[:, 2])), 4)
            except Exception:
                metrics["brier_high_risk"] = np.nan
        else:
            metrics["roc_auc_ovr"] = np.nan
            metrics["pr_auc_high_risk"] = np.nan
            metrics["brier_high_risk"] = np.nan

        return metrics

    @staticmethod
    def business_metrics(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
        portfolio_df: pd.DataFrame,
        false_positive_cost: float,
        false_negative_cost: float,
        collection_cost_per_attempt: float = 15.0,
    ) -> dict[str, float]:
        """Compute business-facing metrics tied to collections economics."""
        recovery_map = {0: 1.0, 1: 0.6, 2: 0.1}

        y_true_s = pd.Series(y_true).reset_index(drop=True)
        y_pred_s = pd.Series(y_pred).reset_index(drop=True)

        actual_recovery_rate = float(y_true_s.map(recovery_map).mean())
        predicted_recovery_rate = float(y_pred_s.map(recovery_map).mean())

        outstanding_total = float(portfolio_df["Outstanding_Balance"].sum())
        collection_attempts = float(portfolio_df["Collection_Attempts"].sum())

        expected_actual_recovery_amount = float((portfolio_df["Outstanding_Balance"] * y_true_s.map(recovery_map).values).sum())
        expected_pred_recovery_amount = float((portfolio_df["Outstanding_Balance"] * y_pred_s.map(recovery_map).values).sum())

        high_true = (y_true_s == 2).astype(int)
        high_pred = (y_pred_s == 2).astype(int)

        tp = int(((high_true == 1) & (high_pred == 1)).sum())
        fn = int(((high_true == 1) & (high_pred == 0)).sum())
        fp = int(((high_true == 0) & (high_pred == 1)).sum())

        high_risk_detection_rate = float(tp / max(1, (high_true == 1).sum()))
        high_risk_precision = float(tp / max(1, (high_pred == 1).sum()))

        collection_efficiency = float(expected_pred_recovery_amount / max(1.0, collection_attempts))

        fp_cost = fp * false_positive_cost
        fn_cost = fn * false_negative_cost
        collection_cost = collection_attempts * collection_cost_per_attempt
        expected_recovery_gain = expected_pred_recovery_amount - expected_actual_recovery_amount
        expected_loss_reduction = max(0.0, expected_actual_recovery_amount - (outstanding_total * 0.1))

        payload = {
            "recovery_rate_actual": round(actual_recovery_rate, 4),
            "recovery_rate_predicted": round(predicted_recovery_rate, 4),
            "expected_actual_recovery_amount": round(expected_actual_recovery_amount, 2),
            "expected_predicted_recovery_amount": round(expected_pred_recovery_amount, 2),
            "expected_recovery_gain": round(expected_recovery_gain, 2),
            "expected_loss_reduction": round(expected_loss_reduction, 2),
            "portfolio_outstanding_total": round(outstanding_total, 2),
            "high_risk_detection_rate": round(high_risk_detection_rate, 4),
            "high_risk_precision": round(high_risk_precision, 4),
            "collection_efficiency": round(collection_efficiency, 2),
            "collection_cost": round(collection_cost, 2),
            "false_positive_cost": round(fp_cost, 2),
            "false_negative_cost": round(fn_cost, 2),
            "total_misclassification_cost": round(fp_cost + fn_cost, 2),
        }

        if y_prob is not None:
            payload["avg_predicted_written_off_probability"] = round(float(np.mean(y_prob[:, 2])), 4)
        else:
            payload["avg_predicted_written_off_probability"] = np.nan

        return payload

    @staticmethod
    def apply_high_risk_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
        """Convert probabilities to class labels using class-2 threshold override."""
        baseline = np.argmax(y_prob, axis=1)
        y_pred = baseline.copy()
        high_mask = y_prob[:, 2] >= threshold
        y_pred[high_mask] = 2

        # For non-high-risk records, choose best class among 0 and 1.
        non_high = ~high_mask
        y_pred[non_high] = np.argmax(y_prob[non_high, :2], axis=1)
        return y_pred

    def optimize_high_risk_threshold(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
        false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        step: float = 0.02,
    ) -> ThresholdOptimizationResult:
        """Optimize class-2 threshold by minimizing cost and tracking precision/recall."""
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        high_true = (y_true_s == 2).astype(int)

        best = ThresholdOptimizationResult(
            threshold=0.5,
            false_positive_cost=np.inf,
            false_negative_cost=np.inf,
            total_cost=np.inf,
            recall_high_risk=0.0,
            precision_high_risk=0.0,
        )
        rows: list[dict[str, float]] = []

        for threshold in np.arange(min_threshold, max_threshold + step / 2, step):
            y_pred = self.apply_high_risk_threshold(y_prob, threshold=threshold)
            high_pred = (pd.Series(y_pred) == 2).astype(int)
            tp = int(((high_true == 1) & (high_pred == 1)).sum())
            fp = int(((high_true == 0) & (high_pred == 1)).sum())
            fn = int(((high_true == 1) & (high_pred == 0)).sum())

            fp_cost = fp * false_positive_cost
            fn_cost = fn * false_negative_cost
            total = fp_cost + fn_cost
            recall = float(tp / max(1, (high_true == 1).sum()))
            precision = float(tp / max(1, (high_pred == 1).sum()))
            rows.append(
                {
                    "threshold": round(float(threshold), 4),
                    "total_cost": round(float(total), 4),
                    "high_risk_recall": round(recall, 4),
                    "high_risk_precision": round(precision, 4),
                    "false_positive_cost": round(float(fp_cost), 4),
                    "false_negative_cost": round(float(fn_cost), 4),
                }
            )
            if total < best.total_cost or (total == best.total_cost and recall > best.recall_high_risk):
                best = ThresholdOptimizationResult(
                    threshold=float(threshold),
                    false_positive_cost=float(fp_cost),
                    false_negative_cost=float(fn_cost),
                    total_cost=float(total),
                    recall_high_risk=recall,
                    precision_high_risk=precision,
                )

        threshold_table = pd.DataFrame(rows)
        threshold_table.to_csv(self.output_dir.parent / "tables" / "high_risk_threshold_sweep.csv", index=False)
        return best

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, filename: str = "confusion_matrix.png") -> Path:
        """Save confusion matrix heatmap."""
        cm = confusion_matrix(pd.Series(y_true).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True))
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_roc_curves(self, y_true: pd.Series, y_prob: np.ndarray, filename: str = "roc_curves.png") -> Path:
        """Save multiclass ROC curve plot."""
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        classes = np.unique(y_true_s)
        y_bin = label_binarize(y_true_s, classes=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, class_id in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {class_id} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title("Multiclass ROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_pr_curve_high_risk(self, y_true: pd.Series, y_prob: np.ndarray, filename: str = "pr_curve_high_risk.png") -> Path:
        """Save precision-recall curve for high-risk (written-off) class."""
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        high_risk_true = (y_true_s == 2).astype(int)
        precision, recall, _ = precision_recall_curve(high_risk_true, y_prob[:, 2])
        pr_auc = average_precision_score(high_risk_true, y_prob[:, 2])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
        ax.set_title("High-Risk Class Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_calibration_high_risk(self, y_true: pd.Series, y_prob: np.ndarray, filename: str = "calibration_high_risk.png") -> Path:
        """Save calibration curve for written-off class probability."""
        y_true_s = pd.Series(y_true).reset_index(drop=True)
        high_risk_true = (y_true_s == 2).astype(int).values
        prob_true, prob_pred = calibration_curve(high_risk_true, y_prob[:, 2], n_bins=10, strategy="quantile")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        ax.set_title("Calibration Curve (Written-Off Class)")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.legend()
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_model_comparison(self, leaderboard: pd.DataFrame, filename: str = "model_comparison.png") -> Path:
        """Save model comparison chart using weighted F1 and ROC-AUC."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.barplot(data=leaderboard, y="model", x="f1_weighted", ax=axes[0], hue="model", legend=False, palette="viridis")
        axes[0].set_title("F1 Weighted by Model")

        roc_df = leaderboard.dropna(subset=["roc_auc_ovr"])
        sns.barplot(data=roc_df, y="model", x="roc_auc_ovr", ax=axes[1], hue="model", legend=False, palette="magma")
        axes[1].set_title("ROC-AUC OVR by Model")

        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

