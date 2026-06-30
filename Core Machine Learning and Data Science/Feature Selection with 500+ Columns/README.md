# Feature Selection with 500+ Columns

Portfolio-grade, production-style project for reducing high-dimensional tabular data into smaller, stronger feature sets while tracking model quality, runtime, memory, and interpretability.

## 1) Problem Statement

High-dimensional datasets (100, 500, 1000+ columns) create practical ML problems:

- curse of dimensionality,
- redundant and noisy predictors,
- overfitting risk,
- higher training/inference cost,
- weaker model interpretability.

This project teaches and implements a professional feature-selection funnel to address those issues end-to-end.

## 2) Dataset Strategy

### Stage 1: Synthetic curriculum

`src/synthetic_generator.py` creates controlled datasets with known:

- informative features,
- redundant features,
- repeated features,
- pure noise features.

This enables ground-truth teaching experiments across 100 / 500 / 1000+ columns.

### Stage 2: Real-world dataset (ISOLET)

- Source: OpenML ISOLET (`did=44010`)
- Rows: 7,797
- Features: 613
- Classes: 26 (spoken letters)
- Missing values: 0 in default source

`src/data_loader.py` caches ISOLET locally under `data/real/isolet.parquet` for reproducible local reruns.

## 3) Feature-Selection Funnel (A to H)

Implemented in `src/feature_selector.py`:

1. Variance Threshold
2. Correlation Filtering (Pearson / Spearman)
3. Model Importance (Random Forest)
4. Permutation Importance
5. RFE / RFECV
6. L1 Regularization
7. Mutual Information
8. SHAP Selection

Reusable APIs:

- `pipeline(...)`
- `fit(...)`
- `transform(...)`
- `fit_transform(...)`

## 4) Benchmarking Stack (Mandatory Tools Included)

Implemented in `src/benchmark.py` and `scripts/generate_benchmark_summary.py`.

### LazyPredict

- Why: rapid baseline ranking.
- Strength: very fast broad sweep.
- Weakness: default hyperparameters, limited control.

### PyCaret

- Why: workflow-level model comparison with less boilerplate.
- Strength: high productivity.
- Weakness: heavier runtime/dependency footprint.
- Note for this project runtime: PyCaret 3.x raises a Python-version guard on Python 3.12, so the benchmark wrapper records `PyCaretUnavailable` rows instead of crashing the pipeline.

### FLAML

- Why: budget-aware hyperparameter optimization.
- Strength: strong quality/time tradeoff.
- Weakness: requires explicit time-budget strategy.

Benchmarks are generated before and after feature selection with unified metrics:

- accuracy, precision, recall, F1, ROC-AUC,
- training time, inference time,
- train/inference peak memory,
- feature-count reduction.

## 5) Notebook Mini-Book

The project includes a zero-to-hero series:

1. `01_synthetic_intro.ipynb`
2. `02_real_dataset_exploration.ipynb`
3. `03_feature_selection_funnel.ipynb`
4. `04_benchmarking.ipynb`
5. `05_advanced_visualizations.ipynb`
6. `06_pipeline_shap_inference.ipynb`
7. `07_error_analysis.ipynb`

Each notebook is written to include:

- concept definition,
- theory + mathematical intuition,
- real-world interpretation,
- visual explanation,
- code explanation,
- result interpretation.

## 6) Reusable Inference Pipeline

`src/inference_pipeline.py` provides a production-oriented wrapper:

- raw feature dataframe input,
- feature-selection funnel execution,
- trained model output,
- selected-feature subset transform,
- persisted artifacts (`model`, `selected_features`, `pipeline_config`, `feature_rankings`).

## 7) Project Structure

```text
Feature Selection with 500+ Columns/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   ├── data_loader.py
│   ├── feature_selector.py
│   ├── benchmark.py
│   ├── inference_pipeline.py
│   ├── synthetic_generator.py
│   └── visualization.py
├── notebooks/
│   ├── 01_synthetic_intro.ipynb
│   ├── 02_real_dataset_exploration.ipynb
│   ├── 03_feature_selection_funnel.ipynb
│   ├── 04_benchmarking.ipynb
│   ├── 05_advanced_visualizations.ipynb
│   ├── 06_pipeline_shap_inference.ipynb
│   └── 07_error_analysis.ipynb
├── scripts/
│   ├── run_all_notebooks.py
│   ├── generate_benchmark_summary.py
│   └── clean_artifacts.py
├── tests/
│   ├── test_feature_selector.py
│   ├── test_synthetic_generator.py
│   ├── test_benchmark_metrics.py
│   ├── test_data_loader.py
│   └── test_inference_pipeline.py
└── outputs/
    ├── figures/
    ├── metrics/
    └── models/
```

## 8) Local Setup (uv + Python 3.12.10)

```bash
cd "Core Machine Learning and Data Science/Feature Selection with 500+ Columns"
uv venv --python 3.12.10 .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra notebooks --extra automl --extra dev
```

## 9) Runbook

### Unit and integration tests

```bash
.venv/bin/python -m pytest -q
```

### Execute all notebooks end-to-end

```bash
.venv/bin/python scripts/run_all_notebooks.py
```

### Regenerate benchmark summary artifacts

```bash
.venv/bin/python scripts/generate_benchmark_summary.py
```

### Clean generated artifacts

```bash
.venv/bin/python scripts/clean_artifacts.py
```

## 10) Key Lessons

- Feature selection must be measured, not assumed beneficial.
- Leakage-safe evaluation boundaries are non-negotiable.
- Different selection techniques expose different signals; funneling is stronger than single-method filtering.
- PCA and feature selection solve different problems:
  - PCA: compact latent representation (less interpretable),
  - Feature selection: keeps original business-facing variables.

## 11) Future Improvements

- Add MLflow/W&B experiment tracking integration.
- Add pandera/Pydantic schema validation at pipeline boundaries.
- Add CI notebook smoke execution.
- Add drift monitoring for selected-feature distributions.
