"""Project package exports for feature-selection workflows."""

from .benchmark import (
    compare_before_after,
    compare_models,
    compute_metrics,
    flaml_optimize,
    lazy_predict_baseline,
    pycaret_compare,
    train_and_evaluate,
)
from .data_loader import load_isolet_dataset
from .feature_selector import FeatureSelector
from .inference_pipeline import FeatureSelectionInferencePipeline, PipelineConfig
from .synthetic_generator import (
    compute_informativeness,
    generate_noise_scale_experiment,
    generate_synthetic_dataset,
)

__all__ = [
    "FeatureSelector",
    "FeatureSelectionInferencePipeline",
    "PipelineConfig",
    "load_isolet_dataset",
    "compute_metrics",
    "train_and_evaluate",
    "compare_models",
    "compare_before_after",
    "lazy_predict_baseline",
    "pycaret_compare",
    "flaml_optimize",
    "generate_synthetic_dataset",
    "compute_informativeness",
    "generate_noise_scale_experiment",
]
