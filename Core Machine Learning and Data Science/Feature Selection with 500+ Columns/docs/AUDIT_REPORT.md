# Repository Audit Report

Date: 2026-06-24

## Scope

Audited:

- Source modules (`src/*.py`)
- Notebooks (`notebooks/01` to `07`)
- Documentation (`README.md`, `docs/architecture.md`)
- Configuration (`pyproject.toml`, `.gitignore`)
- Reproducibility and execution tooling

## Findings and Remediations

| Area | Problem | Why It Matters | Fix Implemented |
|---|---|---|---|
| Synthetic data generator | Informative/redundant/repeated metadata could be wrong due shuffled core features | Invalidates "ground-truth" claims and teaching outcomes | Rebuilt generator: `shuffle=False`, manual repeated-column creation, explicit repeated-source mapping |
| Synthetic repeated features | Repeated columns were corrupted by global noise injection | Repeated features no longer truly repeated; weakens correlation-stage interpretation | Noise now applied before repeated-column cloning, repeated columns stay exact copies |
| Permutation importance | Computed on training data used for fitting | Inflated importance estimates (information leakage) | Added holdout evaluation path (`X_val/y_val` or internal stratified holdout fallback) |
| Variance threshold | Fit done on full `self.X_` while indexing current subset | Potential mask mismatch and incorrect selected mapping | Stage A now fits on current selected subset and maps mask correctly |
| Mutual information | Non-deterministic despite project random state | Results drift run-to-run | Bound `mutual_info_classif(..., random_state=self.random_state)` |
| FeatureSelector API | No reusable fit/transform contract | Harder production reuse and leakage-safe deployment | Added `fit`, `transform`, and `fit_transform` |
| Metrics | Binary-only assumptions in benchmark metrics | Incorrect behavior for multiclass tasks | `compute_metrics` now supports binary + multiclass precision/recall/F1/AUC |
| Benchmark telemetry | No memory metrics | Incomplete benchmarking picture | Added train/inference peak-memory metrics in manual evaluator |
| Notebook drift | Multiple notebooks used outdated FeatureSelector signatures | Broken cells for fresh users | Repaired notebooks and hardened `src/fix_notebooks.py` |
| SHAP notebook stability | SHAP arrays handled inconsistently (list vs 3D) and one collapsed malformed cell | Runtime errors and broken tutorial flow | Rewrote SHAP extraction blocks and repaired malformed cell |
| Path hygiene | Notebook artifacts dumped to `outputs/` root | Messy repo and mixed artifact types | Notebook 06 now writes model artifacts to `outputs/models` and config to `outputs/metrics` |
| Dependency management | All heavy AutoML/notebook deps in base install | Slow/fragile installs for users who only need core pipeline | Moved heavy stacks to optional extras in `pyproject.toml` |
| Reproducibility | No lockfile or one-command execution harness | Hard to reproduce from clone | Added `uv.lock`, `scripts/run_all_notebooks.py`, `scripts/generate_benchmark_summary.py`, `scripts/clean_artifacts.py` |
| Repository cleanliness | Generated caches/egg-info/log artifacts present | Noisy workspace and portability issues | Removed generated caches/artifacts, strengthened `.gitignore` |

## Validation Evidence

- Unit tests: `5 passed` (`.venv/bin/python -m pytest -q`)
- Notebook execution: all 7 notebooks execute via `scripts/run_all_notebooks.py`
- Benchmark artifact generation: `scripts/generate_benchmark_summary.py`
  - `outputs/metrics/benchmark_summary.csv`
  - `outputs/metrics/benchmark_summary.json`

## Residual Limitations

- AutoML wrappers report top-line metrics from third-party tools; full standardized telemetry (including memory for all tools) is partial.
- SHAP execution remains computationally expensive and can vary by hardware/perf budget.
- Notebook outputs are intentionally not committed as default; executed copies are generated under `outputs/executed_notebooks/`.
