import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_package.prediction_engine import PredictionEngine


@pytest.fixture
def trained_model():
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
    ])
    y = np.array([0, 1, 2])
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture
def engine(trained_model):
    return PredictionEngine(trained_model)


class TestPredictionEngine:
    def test_predict_single(self, engine):
        result = engine.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert "prediction" in result
        assert "species" in result
        assert "confidence" in result
        assert "latency_ms" in result
        assert result["model_name"] == "iris_classifier"

    def test_predict_returns_int(self, engine):
        result = engine.predict(np.array([[6.3, 3.3, 6.0, 2.5]]))
        assert isinstance(result["prediction"], int)
        assert 0 <= result["prediction"] <= 2

    def test_predict_known_species(self, engine):
        result = engine.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert result["species"] in ["setosa", "versicolor", "virginica"]

    def test_predict_batch(self, engine):
        features = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
        ])
        results = engine.predict_batch(features)
        assert len(results) == 2
        assert results[0]["sample_id"] == 0
        assert results[1]["sample_id"] == 1

    def test_prediction_has_probabilities(self, engine):
        result = engine.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert result["probabilities"] is not None
        assert len(result["probabilities"]) == 3

    def test_model_info(self, engine):
        info = engine.get_model_info()
        assert info["model_name"] == "iris_classifier"
        assert "n_features_in_" in info
        assert info["target_classes"] == ["setosa", "versicolor", "virginica"]

    def test_without_proba(self, trained_model):
        class NoProbaModel:
            def __init__(self, model):
                self.model = model
            def predict(self, X):
                return self.model.predict(X)

        no_proba = NoProbaModel(trained_model)
        eng = PredictionEngine(no_proba)
        result = eng.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
        assert result["confidence"] is None
        assert result["probabilities"] is None

    def test_predict_negative_values(self, engine):
        with pytest.raises(Exception):
            engine.predict(np.array([[float("nan"), 3.5, 1.4, 0.2]]))
