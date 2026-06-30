"""Core risk-control tests for leakage, metrics, and scenario consistency."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.loan_recovery import FeatureEngineer, LoanDataLoader, ModelEvaluator
from src.loan_recovery.strategy import RecoveryStrategyEngine


class DummyProbModel:
    """Minimal model stub for strategy scenario tests."""

    def __init__(self, feature_names: list[str]) -> None:
        self.named_steps = {
            "preprocess": type("Pre", (), {"feature_names_in_": np.array(feature_names)})(),
        }

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        # Simple deterministic probabilities with valid simplex.
        p2 = np.clip(0.2 + 0.0005 * x["Days_Past_Due"].values + 0.005 * x["Previous_Defaults"].values, 0.01, 0.95)
        p0 = np.clip(0.55 - 0.0003 * x["Days_Past_Due"].values, 0.01, 0.95)
        p1 = 1.0 - p0 - p2
        p1 = np.clip(p1, 0.01, 0.95)
        # Renormalize.
        denom = p0 + p1 + p2
        return np.vstack([p0 / denom, p1 / denom, p2 / denom]).T


class RiskControlsTest(unittest.TestCase):
    """Tests that enforce model-risk and evaluation controls."""

    @classmethod
    def setUpClass(cls) -> None:
        loader = LoanDataLoader("loan-recovery.csv")
        cls.df = loader.load()
        cls.fe = FeatureEngineer()
        cls.enriched = cls.fe.engineer(cls.df)

    def test_leakage_sensitive_features_excluded_by_default(self) -> None:
        x, _ = self.fe.split_features_target(
            self.enriched,
            drop_cols=["Borrower_ID"],
            include_sensitive=False,
            early_warning=True,
        )
        forbidden = {"High_Risk_Flag", "Marital_Status", "Residence_Type", "Collection_Attempts"}
        self.assertTrue(forbidden.isdisjoint(set(x.columns)))

    def test_business_metrics_index_alignment(self) -> None:
        evaluator = ModelEvaluator()
        y_true = pd.Series([2, 2, 0, 1], index=[10, 11, 12, 13])
        y_pred = np.array([2, 2, 0, 1])
        y_prob = np.array(
            [
                [0.05, 0.05, 0.90],
                [0.05, 0.05, 0.90],
                [0.80, 0.15, 0.05],
                [0.10, 0.80, 0.10],
            ]
        )
        portfolio = pd.DataFrame(
            {
                "Outstanding_Balance": [1000, 2000, 1500, 1200],
                "Collection_Attempts": [1, 1, 1, 1],
            }
        )
        metrics = evaluator.business_metrics(y_true, y_pred, y_prob, portfolio, false_positive_cost=100, false_negative_cost=1000)
        self.assertEqual(metrics["high_risk_detection_rate"], 1.0)
        self.assertEqual(metrics["false_negative_cost"], 0.0)

    def test_scenario_table_has_expected_cases(self) -> None:
        model_features = self.fe.model_feature_columns(
            list(self.enriched.columns),
            include_sensitive=False,
            early_warning=True,
            drop_cols=["Borrower_ID", "Recovery_Status"],
        )
        dummy = DummyProbModel(model_features)
        engine = RecoveryStrategyEngine(dummy, self.fe)
        scenarios = engine.what_if_scenarios(self.enriched.head(20))
        expected = {"Baseline", "Income +20%", "EMI -15% (Rate Relief)", "Missed payments +3", "Collateral -20%"}
        self.assertEqual(set(scenarios["scenario"].tolist()), expected)
        self.assertTrue({"avg_risk_score", "avg_recovery_probability", "expected_recovery_value"}.issubset(scenarios.columns))


if __name__ == "__main__":
    unittest.main()

