"""Supervised modeling for loan recovery and high-risk detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .config import CATEGORICAL_COLUMNS, RANDOM_STATE


@dataclass(slots=True)
class ModelArtifacts:
    """Artifacts produced by manual model benchmark."""

    leaderboard: pd.DataFrame
    trained_models: dict[str, Pipeline]
    preprocessor: ColumnTransformer


class ModelTrainer:
    """Train benchmark models and evaluate against held-out data."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.models: dict[str, Pipeline] = {}
        self.leaderboard: pd.DataFrame = pd.DataFrame()
        self.preprocessor: ColumnTransformer | None = None

    def train_baselines(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelArtifacts:
        """Train required benchmark models and compute classification metrics."""
        numeric_cols = [c for c in x_train.columns if c not in CATEGORICAL_COLUMNS]
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in x_train.columns]

        self.preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)

        estimators = self._build_estimators()
        metrics_rows: list[dict[str, float | str]] = []

        for name, estimator in estimators.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", self.preprocessor),
                    ("model", estimator),
                ]
            )
            pipeline.fit(x_train, y_train)
            self.models[name] = pipeline

            y_pred = pipeline.predict(x_test)
            y_prob = pipeline.predict_proba(x_test) if hasattr(pipeline, "predict_proba") else None

            metrics = self._compute_metrics(y_test, y_pred, y_prob)
            metrics_rows.append({"model": name, **metrics})

        self.leaderboard = pd.DataFrame(metrics_rows).sort_values(
            by=["f1_weighted", "roc_auc_ovr"], ascending=[False, False]
        )

        return ModelArtifacts(
            leaderboard=self.leaderboard.reset_index(drop=True),
            trained_models=self.models,
            preprocessor=self.preprocessor,
        )

    def best_model(self, metric: str = "f1_weighted") -> tuple[str, Pipeline]:
        """Return best model name and trained pipeline by metric."""
        if self.leaderboard.empty:
            raise ValueError("No leaderboard found. Train models first.")
        best_name = str(self.leaderboard.iloc[0]["model"] if metric == "f1_weighted" else self.leaderboard.sort_values(metric, ascending=False).iloc[0]["model"])
        return best_name, self.models[best_name]

    def _build_preprocessor(self, numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
        )

    def _build_estimators(self) -> dict[str, Any]:
        estimators: dict[str, Any] = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=180,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=180,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "AdaBoost": AdaBoostClassifier(random_state=self.random_state, n_estimators=120),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state, n_estimators=150),
            "SVM": SVC(
                probability=True,
                random_state=self.random_state,
                class_weight="balanced",
                kernel="linear",
            ),
            "KNN": KNeighborsClassifier(n_neighbors=11),
        }

        # Advanced gradient boosters are imported lazily to keep module import light.
        from xgboost import XGBClassifier

        estimators["XGBoost"] = XGBClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=4,
            tree_method="hist",
            max_bin=256,
            verbosity=0,
        )

        try:
            import lightgbm as lgb

            estimators["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=60,
                learning_rate=0.08,
                max_depth=4,
                num_leaves=15,
                min_child_samples=30,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            )
        except Exception:
            pass

        try:
            from catboost import CatBoostClassifier

            estimators["CatBoost"] = CatBoostClassifier(
                iterations=140,
                depth=5,
                learning_rate=0.05,
                random_state=self.random_state,
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
            )
        except Exception:
            pass

        return estimators

    def imbalance_experiments(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.DataFrame:
        """Compare baseline, class-weighted, and SMOTE variants for imbalance handling."""
        numeric_cols = [c for c in x_train.columns if c not in CATEGORICAL_COLUMNS]
        categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in x_train.columns]
        pre = self._build_preprocessor(numeric_cols, categorical_cols)

        candidates: dict[str, Any] = {
            "LogReg_unweighted": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "LogReg_weighted": LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced"),
            "RF_unweighted": RandomForestClassifier(n_estimators=180, random_state=self.random_state, n_jobs=-1),
            "RF_weighted": RandomForestClassifier(
                n_estimators=180,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
        }
        rows: list[dict[str, float | str]] = []

        for name, estimator in candidates.items():
            pipe = Pipeline([("preprocess", pre), ("model", estimator)])
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_val)
            rows.append(
                {
                    "experiment": name,
                    "f1_weighted": round(float(f1_score(y_val, pred, average="weighted", zero_division=0)), 4),
                    "f1_macro": round(float(f1_score(y_val, pred, average="macro", zero_division=0)), 4),
                    "high_risk_recall": round(float(recall_score((y_val == 2).astype(int), (pred == 2).astype(int), zero_division=0)), 4),
                }
            )

        # Optional SMOTE branch.
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            smote_pipe = ImbPipeline(
                [
                    ("preprocess", pre),
                    ("smote", SMOTE(random_state=self.random_state)),
                    ("model", LogisticRegression(max_iter=2000, random_state=self.random_state)),
                ]
            )
            smote_pipe.fit(x_train, y_train)
            pred = smote_pipe.predict(x_val)
            rows.append(
                {
                    "experiment": "SMOTE_LogReg",
                    "f1_weighted": round(float(f1_score(y_val, pred, average="weighted", zero_division=0)), 4),
                    "f1_macro": round(float(f1_score(y_val, pred, average="macro", zero_division=0)), 4),
                    "high_risk_recall": round(float(recall_score((y_val == 2).astype(int), (pred == 2).astype(int), zero_division=0)), 4),
                }
            )
        except Exception:
            pass

        return pd.DataFrame(rows).sort_values(by=["high_risk_recall", "f1_macro"], ascending=False).reset_index(drop=True)

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        }

        if y_prob is None:
            metrics["roc_auc_ovr"] = np.nan
            metrics["pr_auc_high_risk"] = np.nan
            return metrics

        try:
            metrics["roc_auc_ovr"] = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr")), 4)
        except Exception:
            metrics["roc_auc_ovr"] = np.nan

        # Business-critical class: Written Off (class=2).
        try:
            high_risk_true = (y_true == 2).astype(int)
            high_risk_prob = y_prob[:, 2]
            metrics["pr_auc_high_risk"] = round(float(average_precision_score(high_risk_true, high_risk_prob)), 4)
        except Exception:
            metrics["pr_auc_high_risk"] = np.nan

        return metrics
