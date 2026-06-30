"""LazyPredict benchmark wrapper for fast baseline scanning."""

from __future__ import annotations

import pandas as pd


class LazyPredictBenchmark:
    """Run LazyPredict classification benchmark and expose comparison tables."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.results_: pd.DataFrame = pd.DataFrame()

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Execute LazyClassifier benchmark."""
        from lazypredict.Supervised import CLASSIFIERS, LazyClassifier

        required = {
            "LogisticRegression",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "XGBClassifier",
            "CatBoostClassifier",
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "SVC",
            "KNeighborsClassifier",
        }
        selected_classifiers = [cls for name, cls in CLASSIFIERS if name in required]
        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            random_state=self.random_state,
            classifiers=selected_classifiers,
            timeout=20,
            n_jobs=1,
        )
        models, _ = clf.fit(x_train, x_test, y_train, y_test)
        models = models.reset_index()
        if "index" in models.columns:
            models = models.rename(columns={"index": "model"})
        elif "Model" in models.columns:
            models = models.rename(columns={"Model": "model"})
        else:
            models.insert(0, "model", models.index.astype(str))
        self.results_ = models
        return self.results_

    def top_models(self, n: int = 10, metric: str = "F1 Score") -> pd.DataFrame:
        """Return top-n models sorted by metric."""
        if self.results_.empty:
            return pd.DataFrame()
        ascending = metric == "Time Taken"
        return self.results_.sort_values(metric, ascending=ascending).head(n).reset_index(drop=True)

    def required_model_snapshot(self) -> pd.DataFrame:
        """Return table focused on required model families when available."""
        if self.results_.empty:
            return pd.DataFrame()

        required_patterns = [
            "Logistic",
            "RandomForest",
            "ExtraTrees",
            "XGB",
            "LGBM",
            "CatBoost",
            "AdaBoost",
            "GradientBoosting",
            "SVC",
            "KNeighbors",
        ]

        if "model" not in self.results_.columns:
            if "Model" in self.results_.columns:
                self.results_ = self.results_.rename(columns={"Model": "model"})
            else:
                return pd.DataFrame()

        mask = pd.Series(False, index=self.results_.index)
        for pattern in required_patterns:
            mask = mask | self.results_["model"].str.contains(pattern, case=False, regex=False)

        return self.results_[mask].sort_values("F1 Score", ascending=False).reset_index(drop=True)
