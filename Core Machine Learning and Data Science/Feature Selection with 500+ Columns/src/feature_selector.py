"""
feature_selector.py
-------------------
Reusable feature-selection pipeline implementing the full funnel:

  A. Variance Threshold       - remove low-variance (near-constant) features
  B. Correlation Filtering    - remove highly correlated redundant features
  C. Model-Based Importance   - tree-based feature importance (Random Forest)
  D. Permutation Importance   - model-agnostic importance via shuffling
  E. Recursive Feature Elimination (RFE / RFECV)
  F. L1 Regularization        - Lasso-based sparse selection
  G. Mutual Information       - information-theoretic feature ranking
  H. SHAP-Based Selection     - explainability-driven selection

Every stage is a separate method on FeatureSelector.
A combined pipeline() method runs all stages sequentially.

Design decisions:
  - Each method returns self (fluent API) for chaining.
  - Results are stored in self.results_ dict for inspection.
  - All thresholds are explicit parameters (no hidden defaults).

Inputs:
  - X: pd.DataFrame, feature matrix
  - y: pd.Series, target vector

Outputs:
  - selected_features_: list of chosen feature names after pipeline run
  - results_: dict with per-stage results
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """
    Fluent feature-selection pipeline.

    Usage:
        selector = FeatureSelector()
        selector
            .variance_threshold(threshold=0.01)
            .correlation_filter(threshold=0.95)
            .model_importance(n_estimators=100)
            .permutation_importance()
            .rfe(n_features_to_select=30)
            .l1_selection(C=0.1)
            .mutual_information(k=50)
            .shap_selection(k=30)
        print(selector.selected_features_)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results_: Dict[str, dict] = {}
        self.selected_features_: List[str] = []
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None

    def _check_data(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ):
        """
        Validate and store input data.

        Raises ValueError if X is empty, y is empty, or lengths mismatch.
        y is optional — methods without a y parameter pass None.
        """
        if X.shape[0] == 0:
            raise ValueError("X has 0 rows")
        if y is not None:
            if len(y) == 0:
                raise ValueError("y has 0 elements")
            if X.shape[0] != len(y):
                raise ValueError(
                    f"X rows ({X.shape[0]}) != y length ({len(y)})"
                )
            self.y_ = y
        if self.X_ is None or list(self.X_.columns) != list(X.columns):
            self.selected_features_ = list(X.columns)
        self.X_ = X
        return self

    # ------------------------------------------------------------------
    # A. Variance Threshold
    # ------------------------------------------------------------------
    def variance_threshold(
        self, X: Optional[pd.DataFrame] = None, threshold: float = 0.01
    ) -> "FeatureSelector":
        """
        Remove features with variance below threshold.

        Theory:
          Features with near-zero variance carry almost no information.
          A constant feature (variance=0) is useless for any model.
          We remove features whose variance < threshold.

        Parameters
        ----------
        threshold : float
            Minimum variance to keep a feature. Default 0.01.
            Lower threshold = fewer features removed.
            Common values: 0.0, 0.001, 0.01, 0.05.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = self.y_
        self._check_data(X, y)

        current_features = list(self.selected_features_)
        if not current_features:
            raise ValueError("No features available for variance filtering")
        X_current = self.X_[current_features]

        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X_current)

        # Get mask of kept features
        kept_mask = vt.get_support()
        kept_features = list(np.array(current_features)[kept_mask])
        removed_features = list(
            np.array(current_features)[~kept_mask]
        )

        self.selected_features_ = kept_features
        self.results_["variance_threshold"] = {
            "threshold": threshold,
            "features_in": len(kept_features) + len(removed_features),
            "features_removed": len(removed_features),
            "features_kept": len(kept_features),
            "removed_features": removed_features[:20],
            "variances": pd.Series(
                vt.variances_, index=current_features
            ).to_dict() if hasattr(vt, "variances_") else {},
        }
        return self

    # ------------------------------------------------------------------
    # B. Correlation Filtering
    # ------------------------------------------------------------------
    def correlation_filter(
        self,
        X: Optional[pd.DataFrame] = None,
        threshold: float = 0.95,
        method: str = "pearson",
    ) -> "FeatureSelector":
        """
        Remove features with pairwise correlation above threshold.

        Theory:
          When two features are highly correlated, they carry redundant
          information. Keeping both increases multicollinearity and noise
          without adding signal. We keep the feature with the higher
          variance (more information).

        Parameters
        ----------
        threshold : float
            Maximum absolute correlation allowed. Default 0.95.
            Common values: 0.80, 0.90, 0.95, 0.99.
        method : str
            'pearson' (linear) or 'spearman' (monotonic).

        Returns
        -------
        self
        """
        if method not in ("pearson", "spearman"):
            raise ValueError(f"method must be 'pearson' or 'spearman', got {method}")

        X = X if X is not None else self.X_
        y = self.y_
        self._check_data(X, y)

        # Compute correlation matrix on current selected features
        corr_matrix = (
            self.X_[self.selected_features_].corr(method=method).abs()
        )

        # Upper triangle mask
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find feature pairs above threshold
        to_drop = set()
        for col in upper.columns:
            if col in to_drop:
                continue
            high_corr = list(upper.index[upper[col] > threshold])
            for hc in high_corr:
                # Keep the feature with higher variance
                if col in self.selected_features_ and hc in self.selected_features_:
                    var_col = self.X_[col].var()
                    var_hc = self.X_[hc].var()
                    if var_hc >= var_col:
                        to_drop.add(col)
                    else:
                        to_drop.add(hc)

        kept = [f for f in self.selected_features_ if f not in to_drop]
        self.results_["correlation_filter"] = {
            "threshold": threshold,
            "method": method,
            "features_in": len(self.selected_features_),
            "features_removed": len(to_drop),
            "features_kept": len(kept),
            "removed_features": sorted(list(to_drop))[:20],
        }
        self.selected_features_ = kept
        return self

    # ------------------------------------------------------------------
    # C. Model-Based Importance (Random Forest)
    # ------------------------------------------------------------------
    def model_importance(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        n_estimators: int = 100,
        max_features: Optional[str] = "sqrt",
    ) -> "FeatureSelector":
        """
        Rank and select features using Random Forest feature importance.

        Theory:
          Tree-based models measure how much each feature reduces impurity
          (Gini / entropy) across all splits. Features used higher in the
          tree and more often get higher importance scores. This is a
          fast, reliable embedded method.

        Parameters
        ----------
        n_estimators : int
            Number of trees.
        max_features : str or None
            Max features per split ('sqrt', 'log2', None).

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = y if y is not None else self.y_
        self._check_data(X, y)

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(self.X_[self.selected_features_], self.y_)

        importances = pd.DataFrame({
            "feature": self.selected_features_,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.results_["model_importance"] = {
            "model": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "feature_importances": importances,
        }
        return self

    # ------------------------------------------------------------------
    # D. Permutation Importance
    # ------------------------------------------------------------------
    def permutation_importance(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
        model: Optional[BaseEstimator] = None,
        n_repeats: int = 10,
        n_features_to_show: int = 50,
        holdout_size: float = 0.25,
    ) -> "FeatureSelector":
        """
        Compute permutation importance using a trained model.

        Theory:
          Permutation importance measures how much model score drops when
          a feature's values are randomly shuffled. If shuffling doesn't
          change score, the feature isn't important. This is model-agnostic
          and more reliable than built-in importance for some models.

        Parameters
        ----------
        model : BaseEstimator or None
            Trained model. If None, trains a Random Forest.
        X_eval, y_eval : pd.DataFrame / pd.Series or None
            Holdout data used for permutation scoring. If None, a
            stratified holdout split is created from X/y.
        n_repeats : int
            Number of shuffles per feature.
        n_features_to_show : int
            Number of top features to report.
        holdout_size : float
            Holdout fraction used only when X_eval/y_eval are not provided.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = y if y is not None else self.y_
        self._check_data(X, y)
        X_current = self.X_[self.selected_features_]
        y_current = self.y_

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        evaluation_scope = "provided_holdout"
        if X_eval is None or y_eval is None:
            holdout_size = float(np.clip(holdout_size, 0.1, 0.5))
            try:
                X_fit, X_eval_data, y_fit, y_eval_data = train_test_split(
                    X_current,
                    y_current,
                    test_size=holdout_size,
                    random_state=self.random_state,
                    stratify=y_current,
                )
                evaluation_scope = "internal_holdout"
            except ValueError:
                warnings.warn(
                    "Permutation importance fallback: using training data "
                    "because internal holdout split failed.",
                    RuntimeWarning,
                )
                X_fit, X_eval_data, y_fit, y_eval_data = (
                    X_current,
                    X_current,
                    y_current,
                    y_current,
                )
                evaluation_scope = "train_fallback"
        else:
            X_eval_data = X_eval[self.selected_features_]
            y_eval_data = y_eval
            X_fit, y_fit = X_current, y_current

        model.fit(X_fit, y_fit)

        result = permutation_importance(
            model,
            X_eval_data,
            y_eval_data,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )

        imp_df = pd.DataFrame({
            "feature": self.selected_features_,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

        self.results_["permutation_importance"] = {
            "model": model.__class__.__name__,
            "n_repeats": n_repeats,
            "evaluation_scope": evaluation_scope,
            "fit_rows": int(len(X_fit)),
            "eval_rows": int(len(X_eval_data)),
            "top_features": imp_df.head(n_features_to_show),
        }
        return self

    # ------------------------------------------------------------------
    # E. Recursive Feature Elimination (RFE / RFECV)
    # ------------------------------------------------------------------
    def rfe(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        n_features_to_select: int = 30,
        step: int = 1,
        model: Optional[BaseEstimator] = None,
        use_cv: bool = True,
        min_features_to_select: int = 5,
    ) -> "FeatureSelector":
        """
        Recursively eliminate features by training a model and removing
        the least important features at each step.

        Theory:
          RFE fits a model, ranks features by importance, drops the
          weakest, and repeats. RFECV uses cross-validation to find
          the optimal number of features automatically.

        Parameters
        ----------
        n_features_to_select : int
            Target number of features (RFE) or CV optimal (RFECV uses this
            as a starting point).
        step : int
            Number of features to remove at each iteration.
        model : BaseEstimator or None
            Estimator with coef_ or feature_importances_. Default: RF.
        use_cv : bool
            If True, use RFECV (auto-finds optimal count).
        min_features_to_select : int
            Minimum features for RFECV.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = y if y is not None else self.y_
        self._check_data(X, y)

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        if use_cv:
            cv = StratifiedKFold(3, shuffle=True, random_state=self.random_state)
            selector = RFECV(
                model,
                step=step,
                cv=cv,
                min_features_to_select=min_features_to_select,
                n_jobs=-1,
            )
        else:
            selector = RFE(
                model,
                n_features_to_select=n_features_to_select,
                step=step,
            )

        selector.fit(self.X_[self.selected_features_], self.y_)
        kept_mask = selector.get_support()
        kept = list(np.array(self.selected_features_)[kept_mask])

        self.selected_features_ = kept
        self.results_["rfe"] = {
            "method": "RFECV" if use_cv else "RFE",
            "model": model.__class__.__name__,
            "features_in": len(kept_mask),
            "features_kept": len(kept),
            "n_features_optimal": (
                selector.n_features_ if use_cv else n_features_to_select
            ),
            "ranking": dict(
                zip(self.selected_features_, selector.ranking_[kept_mask])
            )
            if not use_cv
            else {},
        }
        return self

    # ------------------------------------------------------------------
    # F. L1 Regularization (Lasso)
    # ------------------------------------------------------------------
    def l1_selection(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        C: float = 1.0,
        penalty: str = "l1",
        solver: str = "saga",
        max_iter: int = 10000,
    ) -> "FeatureSelector":
        """
        Use L1-regularized logistic regression to select features.

        Theory:
          L1 regularization (Lasso) adds a penalty equal to |coef|.
          This forces some feature coefficients to exactly zero, effectively
          performing feature selection. Features with non-zero coefficients
          are "selected." The C parameter controls regularization strength
          (lower = more regularization = fewer features).

        Parameters
        ----------
        C : float
            Inverse regularization strength. Lower = more regularization.
        penalty : str
            'l1' for Lasso (default). 'l2' for Ridge (no selection).
        solver : str
            'saga' supports L1 and scales to many features.
        max_iter : int
            Maximum solver iterations.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = y if y is not None else self.y_
        self._check_data(X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_[self.selected_features_])
        lr = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lr.fit(X_scaled, self.y_)

        coef_matrix = np.asarray(lr.coef_)
        if coef_matrix.ndim == 1:
            coef_matrix = coef_matrix.reshape(1, -1)

        # Use mean absolute magnitude across classes for multiclass safety.
        coef_strength = np.mean(np.abs(coef_matrix), axis=0)
        mean_signed_coef = np.mean(coef_matrix, axis=0)
        non_zero_mask = coef_strength > 1e-8
        kept = list(np.array(self.selected_features_)[non_zero_mask])

        coef_df = pd.DataFrame({
            "feature": self.selected_features_,
            "coefficient": mean_signed_coef,
            "coefficient_strength": coef_strength,
        }).sort_values("coefficient_strength", ascending=False)

        self.results_["l1_selection"] = {
            "model": "LogisticRegression(L1)",
            "C": C,
            "features_in": len(self.selected_features_),
            "features_kept": len(kept),
            "non_zero_coefficients": coef_df[
                coef_df["coefficient_strength"] > 1e-8
            ],
        }
        self.selected_features_ = kept
        return self

    # ------------------------------------------------------------------
    # G. Mutual Information
    # ------------------------------------------------------------------
    def mutual_information(
        self,
        X: Optional[pd.DataFrame] = None,
        k: int = 50,
    ) -> "FeatureSelector":
        """
        Select top K features by mutual information with the target.

        Theory:
          Mutual information (MI) measures how much knowing a feature
          reduces uncertainty about the target. Unlike correlation, MI
          captures non-linear relationships. MI(X,Y) = H(X) - H(X|Y)
          where H is entropy. Higher MI = more informative.

        Parameters
        ----------
        k : int
            Number of top features to keep.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = self.y_
        self._check_data(X, y)

        selector = SelectKBest(
            score_func=lambda X_arr, y_arr: mutual_info_classif(
                X_arr, y_arr, random_state=self.random_state
            ),
            k=min(k, len(self.selected_features_)),
        )
        selector.fit(self.X_[self.selected_features_], self.y_)

        kept_mask = selector.get_support()
        kept = list(np.array(self.selected_features_)[kept_mask])

        mi_scores = pd.DataFrame({
            "feature": self.selected_features_,
            "mutual_information": selector.scores_,
        }).sort_values("mutual_information", ascending=False)

        self.results_["mutual_information"] = {
            "k": k,
            "features_in": len(self.selected_features_),
            "features_kept": len(kept),
            "top_features": mi_scores.head(k),
        }
        self.selected_features_ = kept
        return self

    # ------------------------------------------------------------------
    # H. SHAP-Based Selection
    # ------------------------------------------------------------------
    def shap_selection(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        k: int = 30,
        model: Optional[BaseEstimator] = None,
        background_samples: int = 50,
    ) -> "FeatureSelector":
        """
        Select top K features by mean absolute SHAP value.

        Theory:
          SHAP (SHapley Additive exPlanations) uses game theory to
          assign each feature a contribution to every prediction.
          Mean |SHAP| across all samples gives a fair, model-agnostic
          measure of global feature importance that accounts for
          feature interactions.

        Parameters
        ----------
        k : int
            Number of top features to keep.
        model : BaseEstimator or None
            Trained model. If None, trains XGBoost.
        background_samples : int
            Number of samples for SHAP background distribution.

        Returns
        -------
        self
        """
        X = X if X is not None else self.X_
        y = y if y is not None else self.y_
        self._check_data(X, y)

        if model is None:
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        # Train model on a subset if large
        model.fit(self.X_[self.selected_features_], self.y_)

        # Use TreeExplainer for tree models
        explainer = shap.TreeExplainer(model)
        # Sample background for speed
        if len(self.X_) > background_samples:
            X_bg = self.X_[self.selected_features_].sample(
                n=background_samples, random_state=self.random_state
            )
        else:
            X_bg = self.X_[self.selected_features_]

        shap_values = explainer.shap_values(X_bg)

        # Multiclass-safe SHAP aggregation: average |SHAP| across samples/classes.
        if isinstance(shap_values, list):
            shap_stack = np.stack([np.asarray(v) for v in shap_values], axis=-1)
            mean_abs_shap = np.mean(np.abs(shap_stack), axis=(0, 2))
        elif getattr(shap_values, "ndim", 0) == 3:
            mean_abs_shap = np.mean(np.abs(np.asarray(shap_values)), axis=(0, 2))
        else:
            mean_abs_shap = np.mean(np.abs(np.asarray(shap_values)), axis=0)

        shap_df = pd.DataFrame({
            "feature": self.selected_features_,
            "mean_abs_SHAP": mean_abs_shap,
        }).sort_values("mean_abs_SHAP", ascending=False)

        top_k = shap_df.head(k)["feature"].tolist()

        self.results_["shap_selection"] = {
            "model": model.__class__.__name__,
            "k": k,
            "features_in": len(self.selected_features_),
            "features_kept": len(top_k),
            "top_features": shap_df.head(k),
        }
        self.selected_features_ = top_k
        return self

    # ------------------------------------------------------------------
    # Combined Pipeline
    # ------------------------------------------------------------------
    def pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        var_threshold: float = 0.01,
        corr_threshold: float = 0.95,
        corr_method: str = "pearson",
        mi_k: int = 120,
        rfe_feat: int = 120,
        rfe_step: int = 5,
        rfe_use_cv: bool = True,
        rfe_min_features: int = 10,
        l1_C: float = 1.0,
        shap_k: int = 120,
        verbose: bool = True,
    ) -> List[str]:
        """
        Run the full feature-selection funnel.

        Order:
          1. VarianceThreshold (remove near-constant)
          2. CorrelationFilter (remove redundant)
          3. MutualInformation (non-linear ranking)
          4. RFE (recursive elimination)
          5. L1 (sparse selection)
          6. SHAP (explainability ranking)

        Model-based importance and permutation importance are computed
        but not used for filtering (they inform interpretation).

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Target vector.
        X_val, y_val : Optional[pd.DataFrame], Optional[pd.Series]
            Optional holdout set used by permutation importance.
        var_threshold : float
            Variance threshold.
        corr_threshold : float
            Correlation threshold.
        corr_method : str
            'pearson' or 'spearman'.
        mi_k : int
            Top-K for mutual information.
        rfe_feat : int
            Target features for RFE.
        rfe_step : int
            Number of features removed per RFE/RFECV iteration.
        rfe_use_cv : bool
            If True, use RFECV. If False, use fixed-count RFE.
        rfe_min_features : int
            Minimum number of features when RFECV is enabled.
        l1_C : float
            L1 regularization strength.
        shap_k : int
            Top-K for SHAP.
        verbose : bool
            If True, print stage progress.

        Returns
        -------
        List[str]
            Final list of selected feature names.
        """
        # Reset stage artifacts for repeatable full-pipeline runs.
        self.results_ = {}
        self.X_ = None
        self.y_ = None
        self.selected_features_ = []

        self._check_data(X, y)
        initial_count = len(self.selected_features_)
        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be provided together")

        if X_val is not None:
            missing = [c for c in self.selected_features_ if c not in X_val.columns]
            if missing:
                raise ValueError(
                    "X_val is missing features required by selector: "
                    f"{missing[:5]}"
                )

        def _log(msg: str):
            if verbose:
                print(msg)

        _log(f"[Pipeline] Starting with {initial_count} features")
        _log(f"[Pipeline] Stage A: VarianceThreshold (>{var_threshold})")
        self.variance_threshold(threshold=var_threshold, X=X)
        _log(f"  -> {len(self.selected_features_)} features remain")

        _log(f"[Pipeline] Stage B: CorrelationFilter (|r|<{corr_threshold})")
        self.correlation_filter(threshold=corr_threshold, method=corr_method)
        _log(f"  -> {len(self.selected_features_)} features remain")

        _log("[Pipeline] Stage C: Model Importance (RF)")
        self.model_importance()

        _log("[Pipeline] Stage D: Permutation Importance")
        self.permutation_importance(
            n_features_to_show=30,
            X_eval=X_val,
            y_eval=y_val,
        )

        _log(f"[Pipeline] Stage E: RFE (target={rfe_feat}, step={rfe_step})")
        self.rfe(
            n_features_to_select=rfe_feat,
            step=rfe_step,
            use_cv=rfe_use_cv,
            min_features_to_select=rfe_min_features,
            X=X,
        )
        _log(f"  -> {len(self.selected_features_)} features remain")

        _log(f"[Pipeline] Stage F: L1 Selection (C={l1_C})")
        self.l1_selection(C=l1_C)
        _log(f"  -> {len(self.selected_features_)} features remain")

        _log(f"[Pipeline] Stage G: Mutual Information (k={mi_k})")
        self.mutual_information(k=mi_k)
        _log(f"  -> {len(self.selected_features_)} features remain")

        _log(f"[Pipeline] Stage H: SHAP Selection (k={shap_k})")
        self.shap_selection(k=shap_k)
        _log(f"  -> {len(self.selected_features_)} features remain")

        reduction = (initial_count - len(self.selected_features_)) / initial_count * 100
        _log(
            f"[Pipeline] Done. Reduced {initial_count} -> "
            f"{len(self.selected_features_)} features "
            f"({reduction:.1f}% reduction)"
        )

        self.results_["pipeline_summary"] = {
            "initial_count": initial_count,
            "final_count": len(self.selected_features_),
            "reduction_percent": round(reduction, 2),
            "stages_completed": [
                "variance_threshold",
                "correlation_filter",
                "model_importance",
                "permutation_importance",
                "rfe",
                "l1_selection",
                "mutual_information",
                "shap_selection",
            ],
        }

        return self.selected_features_

    def fit(self, X: pd.DataFrame, y: pd.Series, **pipeline_kwargs) -> "FeatureSelector":
        """Fit the full feature-selection pipeline on training data."""
        self.pipeline(X=X, y=y, **pipeline_kwargs)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project data to the selected feature subset."""
        if not self.selected_features_:
            raise ValueError("No selected features found. Call fit/pipeline first.")
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(
                "Input is missing selected features: "
                f"{missing[:5]}"
            )
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **pipeline_kwargs
    ) -> pd.DataFrame:
        """Fit selector then transform training features."""
        self.fit(X=X, y=y, **pipeline_kwargs)
        return self.transform(X)
