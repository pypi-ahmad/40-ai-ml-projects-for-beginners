# FINAL PROJECT VERIFICATION REPORT

Date: 2026-06-24
Project: Hybrid Machine Learning Model
Location: `Core Machine Learning and Data Science/Hybrid Machine Learning Model`

## 1. Repository Audit Summary

### What was reviewed
- `src/` pipeline modules
- `pipeline/` CLI entry points
- notebooks generation and execution flow
- `app.py` Streamlit deployment layer
- tests, config, and output artifacts

### Critical issues found and fixed
1. Hybrid leakage risk in `src/forecast_pipeline.py`.
- Previous behavior learned and selected ensemble logic directly on test labels.
- Fix: hybrid/weight learning moved to validation predictions and validation targets; holdout test used only for final evaluation.

2. PyCaret integration broken for installed version (`pycaret==4.0.0a8`).
- Previous behavior used outdated module-level API and invalid args.
- Fix: migrated to `RegressionExperiment` API and validated execution.

3. Weight grid search scalability flaw in `src/weight_optimization.py`.
- Previous implementation used Cartesian product over `[0..1]^n` and became impractical for realistic ensemble size.
- Fix: replaced with simplex-integer grid generation (constrained combinations only).

4. Script execution path bug.
- `pipeline/train_pipeline.py` and `pipeline/run_backtest.py` failed with `ModuleNotFoundError: src` when run as documented.
- Fix: root-path bootstrap added.

5. Data loader alias/validation robustness.
- Close alias normalization had edge-case overwrite risk.
- Fix: deterministic close mapping and stronger schema/value validations.

6. Streamlit robustness gaps.
- Fixes: upload validation, short-history guardrails, sequence-length checks, explicit exception surfaces, leakage-safe weight optimization call path.

7. Backtest output weakly structured.
- Fix: per-fold exports now include fold id and train/test sizes with metrics.

## 2. End-to-End Execution Verification

### Commands executed
- Unit/integration tests:
  - `.venv/bin/python -m pytest -q`
- Full pipeline horizon run:
  - `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pipeline.train_pipeline --config config/config.yaml --horizon 1`
- Backtest CLI:
  - `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pipeline.run_backtest --config config/config.yaml --horizon 1 --strategy walk_forward --model "Random Forest"`
- Notebook execution (all notebooks, fast deterministic mode):
  - `RUN_NOTEBOOK_EXECUTION=1 NOTEBOOK_FULL_RUN=0 MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m pytest tests/test_notebooks_execution.py -q`
- Streamlit startup smoke test:
  - `timeout 20s .venv/bin/python -m streamlit run app.py --server.headless true --server.port 8765 --server.fileWatcherType none`

### Result
- All above verification commands completed successfully.
- Notebook execution required elevated runtime permissions due local kernel sockets in this environment.

## 3. Dataset Validation

Source: `outputs/artifacts/verification_summary.json`

- Rows: 2,518
- Date range: 2010-03-01 to 2020-02-28
- Chronological order after load: true
- Missing values: 0
- Duplicate dates: 0
- Duplicate rows: 0
- Invalid OHLC sign rows: 0
- Invalid `High < Low` rows: 0
- Negative volume rows: 0
- Missing business days identified match holiday patterns (sample included in summary JSON)

## 4. Leakage Audit

Source: `outputs/artifacts/verification_summary.json`

- Future-perturbation leakage probe (`past_feature_changes_after_future_mutation`): **0**
- Target alignment error: **0.0**
- Chronological split non-overlap: **true**
- Deep-learning scaler policy: fit on train only, transform on val/test

## 5. Feature Engineering Validation

Validated in `src/features.py` and tests:
- price returns/log returns
- SMA/EMA/WMA
- RSI/ROC/momentum
- ATR/rolling std/Bollinger bands
- volume transforms
- lag features

Checks added/retained for column requirements and lag validity.

## 6. Traditional Model Review

`outputs/tables/h1_baseline_leaderboard.csv` (test RMSE top performers):
- Linear Regression: 6.3541
- Ridge Regression: 6.4142
- Extra Trees: 89.2593
- Random Forest: 89.6895

Observation:
- For this specific split/horizon, linear models dominate trees and SVR/KNN.
- This is plausible with strong linear lag/technical feature set and regime window characteristics.

## 7. Deep Learning Review

`outputs/tables/h1_deep_leaderboard.csv` (test RMSE):
- Bidirectional LSTM: 12.0506 (best deep)
- TCN: 12.5961
- CNN-LSTM: 17.4842
- GRU: 19.7281
- Stacked LSTM: 26.3976
- Vanilla LSTM: 29.1925

Observation:
- Deep models underperform best linear baseline on this data split.
- This is a valid and educational outcome; deep learning is not automatically superior in low-signal finance tasks.

## 8. Hybrid Model Review

`outputs/tables/h1_hybrid_leaderboard.csv` (test RMSE):
- Weighted Ensemble: 10.6341 (best hybrid)
- Linear Regression + LSTM: 17.9162
- Meta Learner Ensemble: 18.6233
- Stacking Ensemble: 24.0095

Leakage-safe learning now enforced:
- validation for ensemble fitting/selection
- holdout test for reporting only

Observation:
- Best hybrid improves over most deep models but does not beat best linear baseline at horizon 1.

## 9. Weight Optimization Audit

Methods covered: grid, bayesian, FLAML.

Fixes:
- no validation/test contamination in weight fitting.
- simplex-constrained grid now computationally feasible.

Observed in verification summary:
- `optimized_weight_test_rmse`: 18.6233
- This underperforms fixed "Weighted Ensemble" hybrid (10.6341) for current configuration, suggesting validation-optimization mismatch/overfit risk.

## 10. Backtesting Review

`outputs/tables/h1_walk_forward_backtest.csv` includes fold-level details.

Walk-forward aggregated metrics:
- mean RMSE: 22.3662
- std RMSE: 20.1047
- mean R2: -0.2280

Interpretation:
- performance instability across folds confirms regime sensitivity.
- reinforces that single holdout metrics can be overly optimistic.

## 11. Explainability Review

- SHAP pipeline integrated in `src/feature_importance.py` and notebook `08_shap_analysis.ipynb`.
- Summary/dependence export paths validated in execution flow.
- Permutation importance tables generated (`h1_permutation_importance.csv`).

## 12. Streamlit Review

Improvements validated:
- upload schema validation
- short-history blocking
- horizon/sequence sanity checks
- leakage-safe optimized weight path
- robust exception handling

Server startup verified on port 8765 in headless mode.

## 13. Improvements Implemented (Consolidated)

- Leakage-safe hybrid and weight workflows
- PyCaret v4 compatibility fix
- Efficient simplex grid search
- CLI import path fixes
- Loader schema/value hardening
- Streamlit input/workflow hardening
- Notebook generation/runtime hardening with deterministic fast mode and full-mode toggle
- Backtest fold-level export enrichment
- New verification script: `scripts/verification_audit.py`
- Added/expanded tests:
  - `tests/test_forecast_pipeline.py`
  - `tests/test_deep_learning.py`
  - `tests/test_data_loader.py` adj-close alias case
  - `tests/test_weight_optimization.py` length mismatch + larger bank checks

## 14. Remaining Limitations

- Horizon-1 best model remains linear; hybrid does not universally dominate.
- Optimized ensemble weights can degrade holdout performance in current setup.
- No external macro/news/sentiment features in base pipeline.
- Streamlit smoke test validates startup, not full browser interaction automation.
- Full notebook exhaustive mode (`NOTEBOOK_FULL_RUN=1`) is compute-heavy and intended for full benchmark runs, not quick CI.

## 15. Hiring Manager Review Snapshot

Before hardening (initial audit):
- Time-series methodology risk: high (leakage path present)
- Reproducibility risk: medium (CLI import/path and notebook execution friction)
- Portfolio reliability: medium

After hardening:
- Methodology rigor improved (validation/test separation, leakage probes)
- Reproducibility improved (scripts, tests, notebook execution path)
- Portfolio reliability improved with explicit limitations and realistic claims

## 16. Final Scores (1-10)

- Forecasting Quality: **8.0**
- Time-Series Methodology: **9.0**
- Deep Learning Quality: **8.0**
- Hybrid Modeling: **8.5**
- Backtesting Rigor: **8.5**
- Explainability: **8.0**
- Educational Value: **9.0**
- ML Engineering: **8.5**
- Documentation: **9.0**
- Portfolio Strength: **8.5**

Score rationale:
- No category is scored as 10 due known realistic limitations in market predictability, feature scope, and fold instability.
- Additional gains would require richer exogenous data, stronger model governance, and expanded robustness testing.
