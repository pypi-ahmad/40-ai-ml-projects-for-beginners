from __future__ import annotations

from pathlib import Path

from modules.data_generator import generate_synthetic_data
from modules.data_loader import load_dataset
from modules.data_validator import run_full_validation
from modules.feature_engineering import build_feature_pipeline
from modules.feature_selector import run_feature_selection
from modules.model_trainer import run_training_pipeline
from modules.settings import load_config


def test_validation_and_feature_pipeline() -> None:
    config = load_config()
    raw_df = load_dataset("raw", config=config)

    report = run_full_validation(raw_df, config=config)
    assert "checks_passed" in report
    assert report["stats"]["rows"] >= 100

    augmented = generate_synthetic_data(
        raw_df=raw_df,
        days_to_generate=14,
        noise_scale=0.05,
        seasonal_scale=0.1,
        random_state=42,
    )
    assert len(augmented) > len(raw_df)
    assert "Synthetic" in augmented.columns

    featured = build_feature_pipeline(augmented, config=config)
    assert "target_next_day" in featured.columns
    assert featured["target_next_day"].isna().sum() == 0


def test_feature_selection_ranking_created() -> None:
    config = load_config()
    raw_df = load_dataset("raw", config=config)
    augmented = generate_synthetic_data(
        raw_df=raw_df,
        days_to_generate=10,
        noise_scale=0.05,
        seasonal_scale=0.1,
        random_state=42,
    )
    featured = build_feature_pipeline(augmented, config=config)
    ranking = run_feature_selection(featured, config=config)

    assert not ranking.empty
    assert "feature" in ranking.columns
    assert "composite_score" in ranking.columns

    ranking_path = Path(config["paths"]["feature_rankings"])
    assert ranking_path.exists()


def test_training_pipeline_handles_optional_automl_tools() -> None:
    """Regression test for sanitized feature names and optional PyCaret path."""
    config = load_config()
    raw_df = load_dataset("raw", config=config)
    augmented = generate_synthetic_data(
        raw_df=raw_df,
        days_to_generate=7,
        noise_scale=0.05,
        seasonal_scale=0.1,
        random_state=42,
    )
    featured = build_feature_pipeline(augmented, config=config)
    ranking = run_feature_selection(featured, config=config)
    selected = ranking.head(10)["feature"].tolist()

    training = run_training_pipeline(featured, selected_features=selected, config=config)
    assert training["model_name"]
    assert "dependency_status" in training
    assert {"lazypredict", "flaml", "pycaret"}.issubset(set(training["dependency_status"].keys()))
    assert all(" " not in col for col in training["X_test"].columns)
