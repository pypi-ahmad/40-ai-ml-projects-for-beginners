from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.feature_importance import FeatureImportanceAnalyzer, permutation_importance



def make_data(n: int = 200):
    X = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
        }
    )
    y = X["f1"] * 2.0 + X["f2"] * 0.5 + np.random.randn(n) * 0.1
    return X, y



def test_permutation_importance_dict():
    X, y = make_data()
    model = LinearRegression().fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=3)
    assert set(imp.keys()) == set(X.columns)



def test_feature_importance_analyzer_outputs():
    X, y = make_data()
    model = RandomForestRegressor(n_estimators=20, random_state=42).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model=model, X_train=X, y_train=y)
    out = analyzer.compute_all()
    assert "permutation" in out
    assert "built_in" in out
    assert len(analyzer.get_top_features(out["permutation"], n=2)) == 2
    assert isinstance(analyzer.summary(), str)
