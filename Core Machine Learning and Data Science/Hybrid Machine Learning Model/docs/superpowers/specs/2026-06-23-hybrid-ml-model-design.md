# Hybrid Machine Learning Model — Design Spec

## Overview

Build a production-quality Hybrid ML Forecasting System demonstrating how multiple models (ML + DL) can be combined to outperform individual models for Apple stock price forecasting. Portfolio-grade project targeting Data Science, ML Engineering, and Quantitative Finance roles.

## Dataset

- **Source:** Apple stock data, 2010–2020 (2518 rows)
- **Features:** Date, Open, High, Low, Close/Last, Volume
- **Target:** Close/Last price (next day / multi-step ahead)

## Execution Strategy

Sequential phase-by-phase:

| Phase | Contents | Deliverables |
|-------|----------|--------------|
| 1 | Setup, EDA, Feature Engineering | `01_eda.ipynb`, `02_feature_engineering.ipynb`, `src/features.py`, `src/data_loader.py` |
| 2 | Baseline Models + Deep Learning | `03_baseline_models.ipynb`, `04_deep_learning.ipynb`, `src/baseline_models.py`, `src/deep_learning.py` |
| 3 | Hybrid Models + Weight Optimization + Backtesting | `05_hybrid_models.ipynb`, `06_weight_optimization.ipynb`, `07_backtesting.ipynb`, `src/hybrid_models.py`, `src/weight_optimization.py`, `src/backtesting.py` |
| 4 | SHAP + Evaluation + Streamlit + README | `08_shap_analysis.ipynb`, `09_evaluation_report.ipynb`, `app.py`, `README.md`, `src/evaluation.py`, `src/visualization.py`, `src/forecast_pipeline.py` |

## Architecture

### Repository Structure

```
Hybrid Machine Learning Model/
├── data/                           # Dataset (symlink or copy)
│   └── apple_stock_data.csv
├── notebooks/
│   ├── 01_eda.ipynb                # EDA, visualizations, stationarity tests
│   ├── 02_feature_engineering.ipynb # 30+ financial indicators
│   ├── 03_baseline_models.ipynb    # 11 baselines + LazyPredict/PyCaret/FLAML
│   ├── 04_deep_learning.ipynb      # 6 DL architectures (PyTorch)
│   ├── 05_hybrid_models.ipynb      # 7 hybrid ensemble strategies
│   ├── 06_weight_optimization.ipynb # Grid/Bayesian/FLAML weight tuning
│   ├── 07_backtesting.ipynb        # Walk-forward/expanding/rolling CV
│   ├── 08_shap_analysis.ipynb      # SHAP explainability
│   └── 09_evaluation_report.ipynb  # Final benchmarking + conclusions
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load, clean, split (time-series safe)
│   ├── features.py                 # All financial feature functions
│   ├── baseline_models.py          # Scikit-learn/LazyPredict wrappers
│   ├── deep_learning.py            # PyTorch LSTM/GRU/CNN-LSTM/TCN
│   ├── hybrid_models.py            # Stacking/weighted/meta-learner ensembles
│   ├── weight_optimization.py      # GridSearch, BayesianOptimization, FLAML
│   ├── backtesting.py              # Walk-forward, expanding, rolling window
│   ├── evaluation.py               # MAE/MSE/RMSE/MAPE/SMAPE/R² + comparison tables
│   ├── visualization.py            # Matplotlib/Plotly chart templates
│   └── forecast_pipeline.py        # Reusable ForecastingFramework class
├── app.py                          # Streamlit dashboard
├── pyproject.toml                  # uv dependency management
├── outputs/                        # Generated figures (PNG) + CSVs
└── README.md                       # Mini-book documentation
```

### Technology Stack

- **Runtime:** Python 3.12.10, uv
- **Baselines:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Automated ML:** LazyPredict, PyCaret, FLAML
- **Deep Learning:** PyTorch (LSTM, GRU, CNN-LSTM, TCN)
- **Ensemble:** Stacking, Weighted ensembles, Meta-learners
- **Optimization:** Grid Search, Bayesian Optimization, FLAML
- **Explainability:** SHAP
- **Backtesting:** Walk-forward, Expanding window, Rolling window
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit

## Notebook Design (Per-Notebook)

### 01_eda.ipynb
- Data overview, types, missing values
- Price trends (Open/High/Low/Close over time)
- Volume analysis
- Daily returns distribution
- Stationarity tests (ADF, KPSS)
- Autocorrelation (ACF/PACF)
- Decomposition (trend/seasonal/residual)
- Volatility clustering
- Key statistical summaries

### 02_feature_engineering.ipynb
- Advanced returns (daily, log, percentage)
- Moving averages (SMA-5/10/20/50, EMA-5/10/20/50, WMA)
- Momentum (RSI, ROC, Momentum)
- Volatility (rolling std, ATR, Bollinger Bands)
- Volume features (change, moving avg)
- Lag features (1/3/5/10/20/60)
- Target construction (next-day, 5-day, 10-day, 30-day ahead)
- Feature selection, correlation analysis
- Train/test split (time-series aware)

### 03_baseline_models.ipynb
- Naive / Moving Average / Linear / Ridge
- Random Forest / Extra Trees
- XGBoost / LightGBM / CatBoost
- SVR / KNN
- LazyPredict auto-benchmark
- PyCaret compare_models
- FLAML auto-ML
- Comparison table + rankings

### 04_deep_learning.ipynb
- Vanilla LSTM
- Stacked LSTM
- Bidirectional LSTM
- GRU
- CNN-LSTM
- Temporal Convolution Network
- Sequence generation (sliding window)
- Training curves, overfitting analysis

### 05_hybrid_models.ipynb
- Hybrid 1: Linear Regression + LSTM
- Hybrid 2: Random Forest + LSTM
- Hybrid 3: XGBoost + GRU
- Hybrid 4: LightGBM + LSTM
- Hybrid 5: Stacking Ensemble (all models)
- Hybrid 6: Weighted Ensemble (all models)
- Hybrid 7: Meta Learner Ensemble
- Comparison tables

### 06_weight_optimization.ipynb
- Fixed weights baseline
- Grid search over weight space
- Bayesian optimization (scikit-optimize)
- FLAML weight optimization
- Comparison: fixed vs optimized

### 07_backtesting.ipynb
- Walk-forward validation (expanding window)
- Expanding window validation
- Rolling window validation
- Performance across regimes
- Stability analysis

### 08_shap_analysis.ipynb
- SHAP summary plot
- SHAP dependence plot
- Feature importance ranking
- Market signal interpretation

### 09_evaluation_report.ipynb
- Combined results across all models
- Forecasting horizon analysis (1/5/10/30 day)
- Error distribution analysis
- Residual diagnostics
- Final conclusions

## Reusable Forecasting Framework

`src/forecast_pipeline.py` — `ForecastingFramework` class:

```python
class ForecastingFramework:
    def __init__(self, data_path: str)
    def preprocess(self) -> pd.DataFrame
    def create_features(self) -> pd.DataFrame
    def train_baselines(self) -> dict
    def train_deep_learning(self) -> dict
    def train_hybrids(self) -> dict
    def optimize_weights(self, method: str) -> dict
    def backtest(self, strategy: str) -> dict
    def explain(self) -> dict
    def forecast(self, horizon: int) -> pd.DataFrame
    def plot_results(self) -> None
```

## Educational Approach

Every notebook section follows:
1. **Definition** — What is this concept?
2. **Theory** — How does it work mathematically?
3. **Intuition** — Why does it matter for finance?
4. **Business Impact** — Real-world application
5. **Code** — Implementation with explanation
6. **Results** — Interpretation of outputs

## Verification Criteria

- All notebooks execute end-to-end without errors
- Real metrics (MAE, RMSE, MAPE) are computed and reported
- Real figures saved to `outputs/`
- Hybrid models outperform best individual models
- Streamlit app launches and loads data
- README is complete and self-contained

## Constraints

- Local execution only (no cloud APIs)
- Apple stock data only (provided CSV)
- CPU-friendly (GPU optional for DL)
- All dependencies via uv/pyproject.toml
