# Architecture

## System Overview

```text
Synthetic Generator (100/500/1000+ cols)   OpenML ISOLET (613 cols)
                 |                                   |
                 +---------------+-------------------+
                                 v
                     FeatureSelector (A-H funnel)
                                 |
                 +---------------+------------------+
                 |                                  |
                 v                                  v
         Benchmark Engine                    Inference Pipeline
 (Manual + LazyPredict/PyCaret/FLAML)   (fit/transform/predict/save)
                 |                                  |
                 +---------------+------------------+
                                 v
                     Figures, Metrics, Artifacts
```

## Core Modules

### `src/data_loader.py`

- `load_isolet_dataset(...)`
- local parquet cache contract under `data/real/`
- metadata JSON creation for reproducibility

### `src/synthetic_generator.py`

- reproducible high-dimensional synthetic data generation
- explicit feature-group metadata: informative/redundant/repeated/noise
- noise-scale experiment helpers for educational sections

### `src/feature_selector.py`

Implements the full funnel:

1. variance threshold
2. correlation filter
3. model importance
4. permutation importance
5. RFE/RFECV
6. L1 selection
7. mutual information
8. SHAP selection

Design conventions:

- fluent APIs for stage chaining,
- `results_` for per-stage diagnostics,
- `fit/transform` interface for reuse in downstream workflows.

### `src/benchmark.py`

- unified metric computation (binary + multiclass),
- before/after comparison helpers,
- wrappers for LazyPredict, PyCaret, FLAML,
- train/inference latency and memory tracking for manual model runs.

### `src/inference_pipeline.py`

- `PipelineConfig` dataclass for deterministic configuration,
- `FeatureSelectionInferencePipeline` for end-to-end local deployment workflow,
- persisted artifacts:
  - model weights (`.joblib`),
  - selected feature list,
  - pipeline config,
  - stage-level ranking metadata.

### `src/visualization.py`

- reusable publication-ready plotting functions for:
  - funnel retention,
  - importance and SHAP charts,
  - dimensionality reduction,
  - benchmark comparisons,
  - learning curves.

## Notebook Architecture

The notebook suite is intentionally modular:

1. synthetic foundations
2. real-data exploration
3. funnel deep dive
4. benchmarking
5. advanced visuals
6. reusable pipeline + inference
7. error analysis

Each notebook executes independently and can be run sequentially via `scripts/run_all_notebooks.py`.

## Reproducibility and Execution

### `scripts/run_all_notebooks.py`

- executes all notebooks non-interactively with explicit timeout,
- stores executed outputs in `outputs/executed_notebooks/`.

### `scripts/generate_benchmark_summary.py`

- generates benchmark artifacts:
  - `outputs/metrics/benchmark_summary.csv`
  - `outputs/metrics/benchmark_summary.json`

### `scripts/clean_artifacts.py`

- clears generated artifacts for clean reruns.

## Data-Leakage Boundaries

- train/test split precedes all supervised selection/evaluation stages,
- permutation importance uses holdout scoring (`X_val`, `y_val`),
- final performance is reported on untouched test data.

## Design Tradeoffs

- **Interpretability vs compression:** feature selection favored over pure latent transforms for stakeholder explainability.
- **Runtime vs depth:** balanced AutoML budgets are used for local reproducibility.
- **Notebook pedagogy vs library reuse:** critical logic lives in `src/` modules, not hidden in notebook-only state.
