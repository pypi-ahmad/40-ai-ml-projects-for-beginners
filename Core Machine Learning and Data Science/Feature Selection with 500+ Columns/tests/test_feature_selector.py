import pandas as pd
from sklearn.datasets import make_classification

from src.feature_selector import FeatureSelector


def _make_df(n_samples=320, n_features=40, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=6,
        random_state=random_state,
    )
    cols = [f"f_{i:03d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def _make_multiclass_df(n_samples=360, n_features=36, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=14,
        n_redundant=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    cols = [f"f_{i:03d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def test_variance_threshold_works_on_current_feature_subset():
    X, y = _make_df()
    selector = FeatureSelector(random_state=42)
    selector._check_data(X, y)

    # Simulate earlier stage that reduced features.
    selector.selected_features_ = selector.selected_features_[:10]
    selector.variance_threshold(threshold=0.0)

    assert 0 < len(selector.selected_features_) <= 10


def test_permutation_importance_uses_holdout_by_default():
    X, y = _make_df(n_samples=500, n_features=60)
    selector = FeatureSelector(random_state=42)
    selector._check_data(X, y)
    selector.model_importance(X=X, y=y)
    selector.permutation_importance(n_repeats=3, n_features_to_show=10)

    info = selector.results_["permutation_importance"]
    assert info["evaluation_scope"] == "internal_holdout"
    assert info["eval_rows"] < len(X)
    assert info["fit_rows"] < len(X)


def test_transform_returns_selected_columns_only():
    X, y = _make_df(n_samples=260, n_features=30)
    selector = FeatureSelector(random_state=42)
    selector.fit(X, y, verbose=False, shap_k=10, mi_k=12, rfe_feat=15)

    Xt = selector.transform(X)
    assert list(Xt.columns) == selector.selected_features_
    assert Xt.shape[0] == X.shape[0]


def test_l1_selection_handles_multiclass_without_shape_errors():
    X, y = _make_multiclass_df()
    selector = FeatureSelector(random_state=42)
    selector._check_data(X, y)
    selector.l1_selection(C=1.0)

    assert 0 < len(selector.selected_features_) <= X.shape[1]


def test_pipeline_resets_feature_space_between_fits_same_columns():
    X, y = _make_df(n_samples=220, n_features=24)
    selector = FeatureSelector(random_state=42)

    selector.fit(X, y, verbose=False, shap_k=8, mi_k=10, rfe_feat=12)
    first_initial = selector.results_["pipeline_summary"]["initial_count"]

    selector.fit(X, y, verbose=False, shap_k=8, mi_k=10, rfe_feat=12)
    second_initial = selector.results_["pipeline_summary"]["initial_count"]

    assert first_initial == X.shape[1]
    assert second_initial == X.shape[1]
