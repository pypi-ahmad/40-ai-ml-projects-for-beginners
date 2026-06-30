import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_package.model_loader import ModelLoader


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


class TestModelLoader:
    def test_load_unsupported_format(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            ModelLoader("model.txt")

    def test_load_nonexistent_file(self):
        loader = ModelLoader("nonexistent.pkl")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_save_and_load_joblib(self, trained_model):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name

        loader = ModelLoader(path)
        loader.save(trained_model)
        assert Path(path).exists()

        loaded = loader.load()
        pred = loaded.predict([[5.1, 3.5, 1.4, 0.2]])
        assert pred[0] in [0, 1, 2]

    def test_save_and_load_pickle(self, trained_model):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        loader = ModelLoader(path)
        loader.save(trained_model)
        assert Path(path).exists()

        loaded = loader.load()
        pred = loaded.predict([[5.1, 3.5, 1.4, 0.2]])
        assert pred[0] in [0, 1, 2]

    def test_get_metadata_before_load(self, trained_model):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        loader = ModelLoader(path)
        loader.save(trained_model)
        meta = loader.get_metadata()
        assert meta["format"] == ".pkl"
        assert meta["size_bytes"] > 0

    def test_model_property_raises_if_not_loaded(self):
        loader = ModelLoader("dummy.pkl")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = loader.model

    def test_supported_formats(self):
        assert ".pkl" in ModelLoader.SUPPORTED_FORMATS
        assert ".joblib" in ModelLoader.SUPPORTED_FORMATS
        assert ".onnx" in ModelLoader.SUPPORTED_FORMATS
