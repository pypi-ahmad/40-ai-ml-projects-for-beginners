from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.inference_pipeline import FeatureSelectionInferencePipeline, PipelineConfig


def _make_dataset(n_samples: int = 300, n_features: int = 36) -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=8,
        random_state=42,
    )
    cols = [f"f_{i:03d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def test_inference_pipeline_fit_evaluate_and_save(tmp_path: Path):
    X, y = _make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    config = PipelineConfig(
        var_threshold=0.0,
        corr_threshold=0.95,
        mi_k=20,
        rfe_feat=18,
        shap_k=12,
    )
    pipeline = FeatureSelectionInferencePipeline(config=config)
    pipeline.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)

    metrics = pipeline.evaluate(X_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(pipeline.selected_features_) > 0

    transformed = pipeline.transform(X_test)
    assert transformed.shape[1] == len(pipeline.selected_features_)

    artifacts = pipeline.save_artifacts(tmp_path)
    assert Path(artifacts["model_path"]).exists()
    assert Path(artifacts["features_path"]).exists()
    assert Path(artifacts["config_path"]).exists()
    assert Path(artifacts["ranking_path"]).exists()
