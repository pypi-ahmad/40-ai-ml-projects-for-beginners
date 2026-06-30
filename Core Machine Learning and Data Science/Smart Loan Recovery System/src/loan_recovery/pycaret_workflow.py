"""PyCaret-based workflow for comparison with manual modeling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class PyCaretArtifacts:
    """Artifacts produced by PyCaret workflow."""

    best_model_name: str
    comparison_table: pd.DataFrame
    leaderboard_table: pd.DataFrame


class PyCaretWorkflow:
    """Run PyCaret experiment with deterministic, bounded-runtime model checks."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def run(self, data: pd.DataFrame, target_col: str) -> PyCaretArtifacts:
        """Execute PyCaret model checks and return a comparison table.

        Uses `create_model` + `predict_model` for each candidate model to avoid
        long-running compare-model sweeps in constrained local environments.
        """
        from pycaret.classification import ClassificationExperiment

        x = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()

        exp = ClassificationExperiment(
            target=target_col,
            session_id=self.random_state,
            fold=3,
            train_size=0.75,
            fold_strategy="stratifiedkfold",
            preprocess=True,
            n_jobs=1,
            verbose=False,
        )
        exp.fit(x, y)

        candidates = [
            ("lr", "LogisticRegression"),
            ("rf", "RandomForestClassifier"),
            ("et", "ExtraTreesClassifier"),
            ("xgboost", "XGBClassifier"),
        ]

        rows: list[dict[str, float | str]] = []
        for model_id, fallback_name in candidates:
            try:
                created = exp.create_model(model_id, cross_validation=False, verbose=False)
                pred = exp.predict_model(created.pipeline, verbose=False)
                metrics_df = pred.metrics.copy()
                if metrics_df.empty:
                    continue
                row = metrics_df.iloc[0].to_dict()
                row["Model"] = str(row.get("Model", fallback_name))
                row["model_id"] = model_id
                rows.append(row)
            except Exception:
                continue

        comparison = pd.DataFrame(rows)
        if comparison.empty:
            return PyCaretArtifacts(best_model_name="Unavailable", comparison_table=comparison, leaderboard_table=comparison)

        sort_col = "F1" if "F1" in comparison.columns else "Accuracy"
        comparison = comparison.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        best_model_name = str(comparison.iloc[0].get("Model", "Unknown"))
        return PyCaretArtifacts(
            best_model_name=best_model_name,
            comparison_table=comparison,
            leaderboard_table=comparison,
        )

