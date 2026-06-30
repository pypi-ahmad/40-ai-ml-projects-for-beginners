import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml_package.explainability import ModelExplainer

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

shap_required = pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")


@pytest.fixture
def trained_model():
    model = LogisticRegression(max_iter=200)
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
    ])
    y = np.array([0, 1, 2])
    model.fit(X, y)
    return model


class TestModelExplainer:
    def test_no_background_data(self, trained_model):
        explainer = ModelExplainer(trained_model)
        result = explainer.explain_single(
            np.array([[5.1, 3.5, 1.4, 0.2]])
        )
        assert "error" in result

    @shap_required
    def test_explain_single_with_background(self, trained_model):
        X_bg = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5],
        ])
        explainer = ModelExplainer(trained_model, X_bg)
        result = explainer.explain_single(
            np.array([[5.1, 3.5, 1.4, 0.2]])
        )
        assert "base_value" in result
        assert "feature_importance" in result
        assert "sepal_length" in result["feature_importance"]

    @shap_required
    def test_feature_names_in_explanation(self, trained_model):
        X_bg = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5],
        ])
        explainer = ModelExplainer(trained_model, X_bg)
        result = explainer.explain_single(
            np.array([[5.1, 3.5, 1.4, 0.2]])
        )
        assert result["feature_names"] == [
            "sepal_length", "sepal_width", "petal_length", "petal_width"
        ]
