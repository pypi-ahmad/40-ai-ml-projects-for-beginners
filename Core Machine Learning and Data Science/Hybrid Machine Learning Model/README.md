# Hybrid Machine Learning Forecasting System

Portfolio-grade end-to-end project for **financial time-series forecasting** with:

- classical ML
- deep learning (PyTorch)
- hybrid ensembles with learned weights
- leakage-safe validation
- walk-forward backtesting
- SHAP explainability
- Streamlit deployment

Dataset: Apple OHLCV daily market data (`data/apple_stock_data.csv`).

## Executive Summary

This repository demonstrates why hybrid models exist for forecasting noisy markets:

- linear models capture stable trend structure
- tree models capture nonlinear interactions
- sequence models capture temporal memory
- ensembles improve robustness by combining complementary errors

The system is built as reusable production code (`src/`) plus tutorial notebooks (`notebooks/`) that teach every step from fundamentals to deployment.

## 1. Time-Series Forecasting Theory (Beginner to Practitioner)

### What is a time series?
A sequence of observations indexed by time. In this project, each row is one trading day.

### Core components
- `Trend`: long-run directional movement.
- `Seasonality`: fixed-frequency repetition (weak in daily stock prices, stronger in some business data).
- `Cyclic behavior`: irregular macro cycles (bull/bear regimes).
- `Noise`: unpredictable variation.
- `Stationarity`: stable distribution over time (prices are usually non-stationary; returns are closer to stationary).
- `Autocorrelation`: dependency between current and lagged values.
- `Lag features`: explicit memory for tabular models.
- `Forecast horizon`: how far ahead to predict (`1`, `5`, `10`, `30` days).

### Why markets are hard to forecast
- volatility clustering
- regime changes
- macro/news shocks
- nonlinearity
- low signal-to-noise ratio

### Why hybrid models
Hybrid models trade off bias and variance by combining different inductive biases. They often improve **stability** more reliably than they improve best-case peak accuracy.

## 2. Dataset and Validation

- File: `data/apple_stock_data.csv`
- Columns used: `Date`, `Open`, `High`, `Low`, `Close/Last` (normalized to `Close`), `Volume`
- Typical range in this dataset: 2010-03-01 to 2020-02-28

Data loader validations (`src/data_loader.py`):
- schema checks
- date parsing
- duplicate date/row checks
- OHLC positivity checks
- `High >= Low` constraint
- chronological sorting

Audit utility:

```bash
.venv/bin/python scripts/verification_audit.py --config config/config.yaml --horizon 1
```

## 3. Financial Feature Engineering

Implemented in `src/features.py`:

- Price: daily return, log return, percentage change
- Trend: SMA, EMA, WMA
- Momentum: RSI, ROC, momentum
- Volatility: rolling std, ATR, Bollinger bands
- Volume: pct change, moving averages, z-score
- Lags: `1, 3, 5, 10, 20, 60`
- Date features: day/week/month/quarter markers
- Price-position features: intraday range, close position, VWAP relation

Design rule: all rolling/lag features are **causal** (past-only windows).

## 4. Models

### Baseline models
- Naive Forecast
- Moving Average
- Linear Regression
- Ridge Regression
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- CatBoost
- SVR
- KNN Regressor

### Deep models (PyTorch)
- Vanilla LSTM
- Stacked LSTM
- Bidirectional LSTM
- GRU
- CNN-LSTM
- TCN

### Hybrid models
- Linear Regression + LSTM
- Random Forest + LSTM
- XGBoost + GRU
- LightGBM + LSTM
- Stacking Ensemble
- Weighted Ensemble
- Meta Learner Ensemble

## 5. AutoML Stack (Mandatory)

| Tool | Purpose | Strength | Limitation |
|---|---|---|---|
| LazyPredict | broad first-pass benchmark | very fast breadth | shallow optimization |
| PyCaret | low-code comparative workflow | standardized experiment tables | heavier runtime/abstraction |
| FLAML | budget-aware optimization | efficient search under time budgets | less notebook-oriented reporting |

PyCaret integration is implemented against v4 `RegressionExperiment` API.

## 6. Leakage Controls and Methodology Guarantees

Key controls implemented:

- chronological train/validation/test split (no random split)
- target built with forward shift (`target = Close[t+h]`)
- deep-learning scalers fit on **train only**
- hybrid/ensemble weights learned on **validation**, evaluated on **test**
- best weight method selected by validation RMSE, not by test

Backtesting (`src/backtesting.py`):
- walk-forward
- expanding window
- rolling window

## 7. Metrics

Reported metrics (`src/evaluation.py`):
- MAE
- MSE
- RMSE
- MAPE
- sMAPE
- R2

Interpretation rule: RMSE is primary optimization metric; MAPE/sMAPE are included for relative error context.

## 8. Explainability

`src/feature_importance.py` includes:
- permutation importance
- model-native importance/coefs (when available)
- SHAP summary/dependence exports

Outputs are saved under `outputs/plots` and `outputs/tables`.

## 9. Project Structure

```text
Hybrid Machine Learning Model/
├── app.py
├── config/config.yaml
├── data/apple_stock_data.csv
├── notebooks/
├── pipeline/
├── scripts/
├── src/
├── tests/
└── outputs/
```

## 10. Setup and Reproducible Execution

```bash
cd "Core Machine Learning and Data Science/Hybrid Machine Learning Model"
uv venv .venv
source .venv/bin/activate
uv sync
```

### Full pipeline

```bash
.venv/bin/python -m pipeline.train_pipeline --config config/config.yaml
```

### Single horizon

```bash
.venv/bin/python -m pipeline.train_pipeline --horizon 1
```

### Backtest only

```bash
.venv/bin/python -m pipeline.run_backtest --horizon 1 --strategy walk_forward --model "Random Forest"
```

### Verification audit

```bash
.venv/bin/python scripts/verification_audit.py --config config/config.yaml --horizon 1
```

### Notebook generation

```bash
.venv/bin/python _gen_notebooks.py
```

### Notebook execution

Fast deterministic mode (default):

```bash
RUN_NOTEBOOK_EXECUTION=1 NOTEBOOK_FULL_RUN=0 .venv/bin/python -m pytest tests/test_notebooks_execution.py -q
```

Fast mode keeps educational flow but reduces compute:
- AutoML notebook blocks are disabled
- deep-learning epochs are reduced
- weight optimization uses grid-only path

Exhaustive notebook mode:

```bash
RUN_NOTEBOOK_EXECUTION=1 NOTEBOOK_FULL_RUN=1 .venv/bin/python -m pytest tests/test_notebooks_execution.py -q
```

### Streamlit app

```bash
.venv/bin/python -m streamlit run app.py
```

## 11. Financial Reality and Limitations

This project does **not** claim guaranteed prediction, alpha certainty, or market-beating performance.

Limitations:
- no macro/event/news features in core pipeline
- single-asset dataset in current release
- point forecasts are uncertain and degrade with horizon
- backtests are informational, not live trading guarantees

## 12. Production Readiness Notes

Implemented:
- modular pipeline architecture
- deterministic seeding
- structured outputs (tables, plots, predictions, artifacts)
- reproducible configuration via YAML
- test suite for core modules and execution checks

Recommended future hardening:
- model registry + lineage tracking
- drift monitoring/alerts
- CI for notebook execution in dedicated environment
- probabilistic/conformal intervals

## 13. Portfolio Use

Suitable for:
- data science and ML engineering portfolios
- quantitative finance interview walkthroughs
- forecasting/backtesting teaching demos

Primary evidence artifacts:
- `outputs/tables/*leaderboard*.csv`
- `outputs/plots/*`
- `outputs/predictions/*`
- `FINAL_PROJECT_VERIFICATION_REPORT.md`
