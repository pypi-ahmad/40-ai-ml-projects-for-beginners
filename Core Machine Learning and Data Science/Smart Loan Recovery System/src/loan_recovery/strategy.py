"""Data-driven recovery strategy recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import STRATEGY_ACTIONS, TARGET_COLUMN
from .features import FeatureEngineer


@dataclass(slots=True)
class StrategyThresholds:
    """Risk tier boundaries learned from portfolio risk distribution."""

    low_cutoff: float
    medium_cutoff: float
    high_cutoff: float


class RecoveryStrategyEngine:
    """Assign recovery actions using model predictions and portfolio-driven thresholds."""

    def __init__(self, model, feature_engineer: FeatureEngineer) -> None:
        self.model = model
        self.feature_engineer = feature_engineer
        self.thresholds_: StrategyThresholds | None = None

    def _prepare_model_input(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """Align scoring frame to model feature contract."""
        model_input = enriched.drop(columns=[TARGET_COLUMN], errors="ignore")
        if hasattr(self.model, "named_steps"):
            preprocess = self.model.named_steps.get("preprocess")
            names = getattr(preprocess, "feature_names_in_", None)
            if names is not None:
                model_input = model_input.reindex(columns=list(names), fill_value=0)
        return model_input

    def score_portfolio(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Create risk and recovery scoring features from model probabilities."""
        enriched = self.feature_engineer.engineer(raw_df, include_target_derivatives=False)
        model_input = self._prepare_model_input(enriched)

        if self.model is None:
            raise ValueError("A trained model is required to score portfolio.")

        prob = self.model.predict_proba(model_input)
        scored = enriched.copy()
        scored["prob_fully_recovered"] = prob[:, 0]
        scored["prob_partially_recovered"] = prob[:, 1]
        scored["prob_written_off"] = prob[:, 2]

        scored["recovery_probability"] = scored["prob_fully_recovered"] + 0.6 * scored["prob_partially_recovered"]
        scored["risk_score"] = 1.0 - scored["recovery_probability"]

        scored["risk_score"] = (
            0.65 * scored["risk_score"]
            + 0.20 * np.clip(scored["Delinquency_Score"], 0, 1)
            + 0.15 * np.clip(scored["Behavioral_Risk_Score"], 0, 1)
        )
        scored["risk_score"] = np.clip(scored["risk_score"], 0.0, 1.0)

        scored["expected_recovery_value"] = scored["Outstanding_Balance"] * scored["recovery_probability"]
        return scored

    def fit_thresholds(self, scored_df: pd.DataFrame) -> StrategyThresholds:
        """Learn risk thresholds from empirical portfolio risk quantiles."""
        low_cutoff = float(scored_df["risk_score"].quantile(0.50))
        medium_cutoff = float(scored_df["risk_score"].quantile(0.75))
        high_cutoff = float(scored_df["risk_score"].quantile(0.90))

        self.thresholds_ = StrategyThresholds(
            low_cutoff=low_cutoff,
            medium_cutoff=medium_cutoff,
            high_cutoff=high_cutoff,
        )
        return self.thresholds_

    def assign_strategies(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """Assign risk tiers and recommended actions."""
        if self.thresholds_ is None:
            self.fit_thresholds(scored_df)

        thresholds = self.thresholds_
        if thresholds is None:
            raise ValueError("Unable to fit thresholds.")

        bins = [-np.inf, thresholds.low_cutoff, thresholds.medium_cutoff, thresholds.high_cutoff, np.inf]
        labels = ["Low", "Medium", "High", "Very High"]

        assigned = scored_df.copy()
        assigned["risk_tier"] = pd.cut(assigned["risk_score"], bins=bins, labels=labels, include_lowest=True)
        assigned["risk_tier"] = assigned["risk_tier"].astype(str)
        assigned["recommended_strategy"] = assigned.apply(self._derive_strategy, axis=1)

        # Prioritization score blends risk and collectible value.
        assigned["priority_score"] = (
            assigned["risk_score"]
            * np.log1p(assigned["Outstanding_Balance"])
            * (1 + np.clip(assigned["Recovery_Difficulty_Index"], 0, 1))
            * (1 + assigned["prob_written_off"])
        )
        assigned["priority_rank"] = assigned["priority_score"].rank(ascending=False, method="dense").astype(int)
        return assigned.sort_values("priority_rank")

    @staticmethod
    def _derive_strategy(row: pd.Series) -> str:
        """Translate risk + borrower profile into operational strategy."""
        tier = row["risk_tier"]
        collateral = float(row.get("Collateral_Coverage_Ratio", 1.0))
        difficulty = float(row.get("Recovery_Difficulty_Index", 0.5))
        defaults = float(row.get("Previous_Defaults", 0.0))
        delinquency = float(row.get("Delinquency_Score", 0.0))

        if tier == "Low":
            return STRATEGY_ACTIONS["Low"]
        if tier == "Medium":
            return STRATEGY_ACTIONS["Medium"]
        if tier == "High":
            if collateral < 0.9 and difficulty > 0.7:
                return STRATEGY_ACTIONS["High"]
            return STRATEGY_ACTIONS["Medium"]
        if tier == "Very High":
            if collateral < 0.8 or defaults >= 2 or delinquency > 0.7:
                return STRATEGY_ACTIONS["Very High"]
            return STRATEGY_ACTIONS["High"]
        return STRATEGY_ACTIONS["Medium"]

    def segment_recommendations(self, assigned_df: pd.DataFrame, segment_col: str = "segment") -> pd.DataFrame:
        """Aggregate strategy guidance by borrower segment."""
        if segment_col not in assigned_df.columns:
            return pd.DataFrame()

        grouped = (
            assigned_df.groupby(segment_col, as_index=False)
            .agg(
                borrowers=("Borrower_ID", "count"),
                avg_risk_score=("risk_score", "mean"),
                avg_recovery_probability=("recovery_probability", "mean"),
                total_outstanding=("Outstanding_Balance", "sum"),
                dominant_strategy=("recommended_strategy", lambda x: x.mode().iloc[0]),
            )
            .sort_values("avg_risk_score", ascending=False)
        )
        return grouped

    def what_if_scenarios(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run scenario analysis to quantify portfolio sensitivity to input changes."""
        scenarios = {
            "Baseline": {},
            "Income +20%": {"Monthly_Income": 1.20},
            "EMI -15% (Rate Relief)": {"Interest_Rate": 0.85},
            "Missed payments +3": {"Missed_Payments": 3.0, "Days_Past_Due": 30.0},
            "Collateral -20%": {"Collateral_Value": 0.80},
        }

        rows: list[dict[str, float | str]] = []
        for name, changes in scenarios.items():
            scenario_df = raw_df.copy()
            for col, value in changes.items():
                if col not in scenario_df.columns:
                    continue
                if col in {"Missed_Payments", "Days_Past_Due"}:
                    scenario_df[col] = np.clip(scenario_df[col] + value, 0, None)
                else:
                    scenario_df[col] = scenario_df[col] * value
                    if col == "Interest_Rate":
                        scenario_df[col] = scenario_df[col].clip(lower=1.0, upper=60.0)

            scored = self.score_portfolio(scenario_df)
            rows.append(
                {
                    "scenario": name,
                    "avg_risk_score": round(float(scored["risk_score"].mean()), 4),
                    "avg_recovery_probability": round(float(scored["recovery_probability"].mean()), 4),
                    "expected_recovery_value": round(float(scored["expected_recovery_value"].sum()), 2),
                }
            )

        return pd.DataFrame(rows)
