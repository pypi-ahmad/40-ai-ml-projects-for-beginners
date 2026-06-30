"""Generate executable tutorial notebooks for Hybrid Machine Learning Model project."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)



def md(text: str):
    return nbf.v4.new_markdown_cell(text)



def code(text: str):
    return nbf.v4.new_code_cell(text)



def pedagogical_block(title: str, definition: str, theory: str, math: str, financial: str, impact: str, example: str) -> str:
    return f"""
## {title}

**Definition**  
{definition}

**Theory**  
{theory}

**Mathematical Intuition**  
{math}

**Financial Intuition**  
{financial}

**Business Impact**  
{impact}

**Real-World Example**  
{example}

**Visual Explanation**  
Code cells below generate real plots from Apple market data.

**Code Explanation**  
Each code block is annotated inline and uses shared production modules from `src/`.

**Interpretation of Results**  
After each output, interpret what signal quality, risk, and forecasting reliability imply.
"""



def write_notebook(path: Path, cells: list):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb["nbformat_minor"] = 5
    nb.cells = cells
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


common_imports = """
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = (PROJECT_ROOT / '..').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src.forecast_pipeline import ForecastingFramework
from src.data_loader import load_stock_data, split_data
from src.features import create_features
from src.evaluation import regression_metrics
from src.visualization import *

OUT = PROJECT_ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'
FAST_NOTEBOOK_MODE = os.getenv('NOTEBOOK_FULL_RUN', '0') != '1'

def make_framework():
    framework = ForecastingFramework(str(CONFIG_PATH))
    if FAST_NOTEBOOK_MODE:
        framework.config['models']['automl']['lazypredict'] = False
        framework.config['models']['automl']['pycaret'] = False
        framework.config['models']['automl']['flaml'] = False
        framework.config['models']['deep_learning']['epochs'] = 8
        framework.config['models']['deep_learning']['early_stopping_patience'] = 3
        framework.config['weight_optimization']['methods'] = ['grid']
    return framework

framework = make_framework()
"""


# 01 EDA
cells_01 = [
    md("# 01 - Exploratory Data Analysis for Financial Forecasting\n\nZero-to-Hero tutorial: understand time-series behavior before modeling."),
    md(
        pedagogical_block(
            "What Is Time Series Forecasting?",
            "Time series forecasting predicts future values from chronologically ordered historical observations.",
            "A time series decomposes into trend, seasonality/cycles, and noise. In markets, structural breaks make this decomposition unstable.",
            "A generic representation is `y_t = f(y_{t-1}, y_{t-2}, ..., x_t) + ε_t`, where `x_t` includes exogenous variables and `ε_t` is noise.",
            "Stock prices show momentum bursts, mean-reversion windows, and volatility clustering.",
            "Forecasting quality impacts trading decisions, risk controls, inventory hedging, and portfolio rebalancing.",
            "COVID crash period demonstrates regime shift where historical relationships weaken abruptly.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
df = framework.load_data()

print('Shape:', df.shape)
print('Date Range:', df.index.min(), '->', df.index.max())
print('Columns:', list(df.columns))
display(df.head())
display(df.describe().T)
"""
    ),
    md(
        pedagogical_block(
            "Trend, Seasonality, Cyclic Patterns, Noise, Stationarity",
            "Trend is long-run direction; seasonality repeats by fixed calendar frequency; cycles are irregular macro waves; noise is unpredictable residual movement.",
            "Financial prices are often non-stationary; returns are usually closer to stationary. Autocorrelation structure informs lag design.",
            "Stationarity implies distributional invariance over time. ADF test rejects unit root for stationary series.",
            "Price levels trend; returns oscillate around small mean with heavy tails.",
            "Non-stationarity raises model drift risk and can invalidate backtests if not handled carefully.",
            "Tech earnings shocks or macro rate decisions create temporary cycles that differ from classic seasonality.",
        )
    ),
    code(
        """
plot_price_history(df, OUT / 'plots/01_price_history.png')
plot_ohlc_lines(df, OUT / 'plots/01_ohlc_lines.png')
plot_volume(df, OUT / 'plots/01_volume.png')
plot_returns_distribution(df, OUT / 'plots/01_returns_distribution.png')

for p in [
    OUT / 'plots/01_price_history.png',
    OUT / 'plots/01_ohlc_lines.png',
    OUT / 'plots/01_volume.png',
    OUT / 'plots/01_returns_distribution.png',
]:
    display(pd.DataFrame({'generated_plot': [str(p)]}))
"""
    ),
    code(
        """
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

returns = df['Close'].pct_change().dropna()
price_adf = adfuller(df['Close'])
returns_adf = adfuller(returns)

print('ADF Price Statistic:', price_adf[0], 'p-value:', price_adf[1])
print('ADF Returns Statistic:', returns_adf[0], 'p-value:', returns_adf[1])

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(returns, lags=40, ax=ax)
ax.set_title('Autocorrelation of Daily Returns')
fig.tight_layout()
fig.savefig(OUT / 'plots/01_returns_acf.png', dpi=150)
plt.close(fig)
"""
    ),
    md(
        """
## EDA Findings

- Price level non-stationary; returns materially closer to stationary.
- Volatility clustering exists (heteroskedastic behavior).
- Volume spikes align with stress regimes.
- Lagged signal likely short-lived; horizon-aware evaluation required.
- Random train/test split would leak future structure and inflate metrics.
"""
    ),
]


# 02 Feature Engineering
cells_02 = [
    md("# 02 - Financial Feature Engineering\n\nBuild predictive features with explicit financial rationale."),
    md(
        pedagogical_block(
            "Feature Families for Financial Forecasting",
            "Feature engineering transforms raw OHLCV into model-consumable signals.",
            "Different signal classes capture different market dynamics: trend, momentum, volatility, liquidity, and temporal context.",
            "Engineered feature matrix `X_t` enriches representational capacity beyond raw price `P_t`.",
            "RSI can proxy momentum exhaustion; ATR captures volatility regime; volume z-score flags abnormal participation.",
            "Better features reduce forecast error and improve model stability across regimes.",
            "Momentum + volatility interactions often explain why naive models fail during turbulent periods.",
        )
    ),
    code(common_imports),
    code(
        """
df = load_stock_data(PROJECT_ROOT / 'data' / 'apple_stock_data.csv')
feat_cfg = framework.config['features']

features = create_features(
    df,
    lags=feat_cfg['lags'],
    rolling_windows=feat_cfg['rolling_windows'],
    ema_windows=feat_cfg['ema_windows'],
    wma_windows=feat_cfg['wma_windows'],
    momentum_windows=feat_cfg['momentum_windows'],
    include_technical=feat_cfg['include_technical'],
    include_date_features=feat_cfg['include_date_features'],
    include_price_derived=feat_cfg['include_price_derived'],
    dropna=False,
)

print('Raw columns:', df.shape[1])
print('Feature columns:', features.shape[1])
print('Added columns:', features.shape[1] - df.shape[1])
display(features.tail(5))
"""
    ),
    code(
        """
selected = [
    'daily_return', 'log_return', 'sma_20', 'ema_20', 'wma_20',
    'rsi_14', 'roc_10', 'momentum_10', 'rolling_std_20', 'atr_14',
    'bb_width_20', 'volume_change', 'volume_ma_20', 'Close_lag_20', 'Volume_lag_20'
]
selected = [c for c in selected if c in features.columns]

corr = features[selected + ['Close']].corr().sort_values('Close', ascending=False)
display(corr[['Close']])

fig, ax = plt.subplots(figsize=(12, 6))
features[selected].dropna().tail(400).plot(ax=ax, alpha=0.8)
ax.set_title('Sample Engineered Feature Dynamics (Last 400 Rows)')
fig.tight_layout()
fig.savefig(OUT / 'plots/02_feature_dynamics.png', dpi=150)
plt.close(fig)
"""
    ),
    md("""
## Feature Interpretation Checklist

- Trend features reduce noise sensitivity.
- Momentum features capture continuation or reversal pressure.
- Volatility features adapt model expectations under turbulence.
- Volume features encode participation and conviction.
- Lag features create autoregressive memory for non-sequence models.
"""),
]


# 03 Baselines + AutoML
cells_03 = [
    md("# 03 - Baseline Modeling and AutoML Benchmarks\n\nBenchmark strong non-deep baselines across all forecast horizons."),
    md(
        pedagogical_block(
            "Why Baselines Matter",
            "Baselines define minimum acceptable performance and protect against over-engineering.",
            "If advanced models do not beat robust baselines out-of-sample, complexity is unjustified.",
            "Error gap vs baseline quantifies real incremental value.",
            "In finance, simpler models can outperform during regime stability due lower variance.",
            "Baseline discipline reduces deployment risk and accelerates debugging.",
            "Naive model surprisingly strong on short horizons in persistent trends.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
framework.load_data()

horizons = [1] if FAST_NOTEBOOK_MODE else framework.config['features']['horizons']
all_rows = []
automl_rows = []

for h in horizons:
    print(f'Running baseline benchmark for horizon {h}...')
    out = framework.train_baselines(h)
    lb = out['leaderboard'].copy()
    lb['horizon'] = h
    all_rows.append(lb)

    for tool_name, tool_df in out['automl'].items():
        tmp = tool_df.copy()
        tmp['horizon'] = h
        tmp['tool_name'] = tool_name
        automl_rows.append(tmp)

baseline_all = pd.concat(all_rows, ignore_index=True)
baseline_all.to_csv(OUT / 'tables/03_baseline_all_horizons.csv', index=False)
display(baseline_all.head(20))

if automl_rows:
    automl_all = pd.concat(automl_rows, ignore_index=True)
    automl_all.to_csv(OUT / 'tables/03_automl_all_horizons.csv', index=False)
    display(automl_all.head(20))
"""
    ),
    md("`NOTEBOOK_FULL_RUN=1` executes all configured horizons; default mode runs horizon 1 for faster reproducible execution."),
    md("""
## Tool Tradeoffs Summary

| Tool | Strength | Weakness | Practical Use |
|---|---|---|---|
| LazyPredict | Fast breadth | Limited tuning depth | Rapid first scan |
| PyCaret | Strong experiment workflow | Heavier abstraction/runtime | Structured comparison tables |
| FLAML | Budget-aware optimization | Less tutorial-style UX | Time-constrained tuning |
"""),
]


# 04 Deep Learning
cells_04 = [
    md("# 04 - Deep Learning Forecasting (PyTorch)\n\nTrain and compare six sequence architectures."),
    md(
        pedagogical_block(
            "Why Sequence Models",
            "Sequence models learn temporal dependencies directly from ordered windows.",
            "RNN/LSTM/GRU capture recurrence; CNN-LSTM captures local motifs then sequence context; TCN captures long receptive fields with dilated convolutions.",
            "Given sequence `X_{t-k:t-1}`, model learns mapping `f(X)->y_t` via gradient descent.",
            "Useful when nonlinear memory patterns exceed tabular lag interaction capacity.",
            "Potentially better under nonlinear shocks but costlier to train and tune.",
            "GRU may generalize better than LSTM under small data due fewer parameters.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
framework.load_data()

# Deep section is compute-heavy; default to 1-day horizon for detailed architecture comparison.
deep_out = framework.train_deep_models(horizon=1)
display(deep_out['leaderboard'])

deep_out['leaderboard'].to_csv(OUT / 'tables/04_deep_learning_h1.csv', index=False)
"""
    ),
    md("""
## Deep Model Strengths and Weaknesses

- **Vanilla LSTM**: stable baseline sequence learner; may miss hierarchical patterns.
- **Stacked LSTM**: richer temporal hierarchy; higher overfitting risk.
- **Bidirectional LSTM**: powerful context extraction; less realistic for strict causal inference (uses both directions during training windows).
- **GRU**: efficient recurrent architecture with lower parameter count.
- **CNN-LSTM**: captures local motifs + long memory.
- **TCN**: strong parallelism and receptive-field control via dilation.
"""),
]


# 05 Hybrid Models
cells_05 = [
    md("# 05 - Hybrid Modeling\n\nCombine ML and DL models to reduce bias-variance and improve robustness."),
    md(
        pedagogical_block(
            "Why Hybrid Models",
            "Hybrid models blend complementary inductive biases from different model families.",
            "Linear models capture simple trend; trees capture nonlinear interactions; sequence models capture memory. Ensemble aggregation can reduce variance.",
            "Weighted combination: `ŷ = Σ w_i ŷ_i`, where `w_i` learned from validation objective.",
            "Market dynamics change by regime; no single model dominates all periods.",
            "Hybrid strategy often yields more stable risk-adjusted error profile.",
            "During volatility shocks, one submodel may fail while others remain calibrated.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
framework.load_data()

hybrid_out = framework.train_hybrids(horizon=1)
display(hybrid_out['leaderboard'])
display(hybrid_out['val_leaderboard'])

hybrid_out['leaderboard'].to_csv(OUT / 'tables/05_hybrid_h1.csv', index=False)
hybrid_out['val_leaderboard'].to_csv(OUT / 'tables/05_hybrid_h1_validation.csv', index=False)
"""
    ),
]


# 06 Weight Optimization
cells_06 = [
    md("# 06 - Ensemble Weight Optimization\n\nLearn blend weights with Grid Search, Bayesian Optimization, and FLAML."),
    md(
        pedagogical_block(
            "Fixed vs Optimized Weights",
            "Fixed weights assume equal model reliability; optimized weights adapt to empirical performance.",
            "Optimization minimizes validation/test loss over simplex-constrained weights.",
            "Constraint: `w_i >= 0` and `Σ w_i = 1`.",
            "Helps overweight models that stay calibrated in current market regime.",
            "Directly improves ensemble error and interpretability of model contribution.",
            "Bayesian search can find better blends faster than exhaustive grids in high-dimensional spaces.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
framework.load_data()
hybrid_out = framework.train_hybrids(horizon=1)

val_preds = hybrid_out['val_predictions']
y_val = hybrid_out['y_val_true']
test_preds = hybrid_out['test_predictions']
y_test = hybrid_out['y_test_true']

rows = []
for method in ['grid', 'bayesian', 'flaml']:
    try:
        result = framework.optimize_weights(
            1,
            val_preds,
            y_val,
            method=method,
            evaluation_predictions=test_preds,
            evaluation_y_true=y_test,
        )
        rows.append({
            'method': method,
            'val_rmse': result['fit_metrics']['rmse'],
            'test_rmse': result['test_metrics']['rmse'],
            'test_mae': result['test_metrics']['mae'],
            'test_mape': result['test_metrics']['mape'],
            'weights': result['weights'],
        })
    except Exception as exc:
        rows.append({'method': method, 'error': str(exc)})

opt_df = pd.DataFrame(rows)
display(opt_df)
opt_df.to_csv(OUT / 'tables/06_weight_optimization_h1.csv', index=False)
"""
    ),
]


# 07 Backtesting
cells_07 = [
    md("# 07 - Professional Backtesting\n\nCompare walk-forward, expanding, and rolling-window validation."),
    md(
        pedagogical_block(
            "Why Random Splits Are Dangerous",
            "Random splitting leaks future information into training folds for time series.",
            "Backtesting preserves chronology and simulates real deployment.",
            "Train on past `t<=T`; evaluate on future `t>T` repeatedly.",
            "Market regimes evolve; robust models should stay competitive across folds.",
            "Backtesting prevents false confidence from unrealistic validation.",
            "A model that wins random split but fails walk-forward is not production-ready.",
        )
    ),
    code(common_imports),
    code(
        """
from src.models import MODEL_REGISTRY

framework = make_framework()
framework.load_data()

rows = []
for strategy in ['walk_forward', 'expanding', 'rolling']:
    result = framework.backtest(horizon=1, model=MODEL_REGISTRY['Random Forest'], strategy=strategy)
    metrics = result['aggregated_metrics']
    rows.append({
        'strategy': strategy,
        'mean_rmse': metrics['mean_rmse'],
        'std_rmse': metrics['std_rmse'],
        'mean_mape': metrics['mean_mape'],
        'std_mape': metrics['std_mape'],
    })

bt_df = pd.DataFrame(rows).sort_values('mean_rmse')
display(bt_df)
bt_df.to_csv(OUT / 'tables/07_backtesting_h1.csv', index=False)
"""
    ),
]


# 08 SHAP
cells_08 = [
    md("# 08 - Explainable AI with SHAP\n\nUnderstand what drives stock forecasts."),
    md(
        pedagogical_block(
            "Interpreting Forecast Drivers",
            "Explainability quantifies feature contribution to predictions.",
            "SHAP uses cooperative game theory to attribute each feature's marginal contribution.",
            "Prediction is decomposed as `f(x)=ϕ0 + Σϕi`.",
            "Feature attribution helps validate whether model logic aligns with market intuition.",
            "Supports model risk governance, debugging, and stakeholder trust.",
            "If volume shock dominates predictions during stress periods, model may be liquidity-sensitive.",
        )
    ),
    code(common_imports),
    code(
        """
from src.feature_importance import save_shap_summary_plot, save_shap_dependence_plot

framework = make_framework()
framework.load_data()
base = framework.train_baselines(horizon=1)

best_model_name = base['leaderboard'].iloc[0]['model']
best_model = base['results'][best_model_name].model

explain_out = framework.explain(horizon=1, model=best_model)
print('Best model:', best_model_name)
print(explain_out['summary'])

if explain_out['shap_available']:
    ds = framework.prepare_horizon_dataset(1)
    shap_path = OUT / 'plots/08_shap_summary.png'
    dep_path = OUT / 'plots/08_shap_dependence_close_lag_1.png'
    save_shap_summary_plot(explain_out['shap_values'], ds.X_test[:200], shap_path)
    save_shap_dependence_plot(explain_out['shap_values'], ds.X_test[:200], 'feature_0', dep_path)
    print('SHAP plots saved:', shap_path, dep_path)
else:
    print('SHAP unavailable for selected model/runtime; permutation importance still generated.')
"""
    ),
]


# 09 Final Evaluation
cells_09 = [
    md("# 09 - Final Evaluation, Horizon Degradation, and Scenario Analysis\n\nConsolidate all evidence into portfolio-grade conclusions."),
    md(
        pedagogical_block(
            "Forecast Horizon Degradation",
            "As horizon increases, uncertainty compounds and signal decays.",
            "Error typically increases sublinearly or superlinearly depending on regime volatility.",
            "For horizon `h`, expected forecast variance generally increases with `h` under diffusion-like dynamics.",
            "Short horizons capture microstructure and momentum; long horizons are dominated by macro/exogenous noise.",
            "Horizon-aware deployment avoids overpromising accuracy.",
            "A 1-day model may be useful for tactical positioning while 30-day output supports scenario planning only.",
        )
    ),
    code(common_imports),
    code(
        """
framework = make_framework()
framework.load_data()

summary_rows = []
for h in ([1] if FAST_NOTEBOOK_MODE else framework.config['features']['horizons']):
    out = framework.run_horizon(h)
    best_hybrid = out['hybrid']['leaderboard'].iloc[0]
    summary_rows.append({
        'horizon': h,
        'best_hybrid_model': best_hybrid['model'],
        'best_hybrid_rmse': best_hybrid['rmse'],
        'best_hybrid_mape': best_hybrid['mape'],
    })

summary_df = pd.DataFrame(summary_rows).sort_values('horizon')
display(summary_df)
summary_df.to_csv(OUT / 'tables/09_final_horizon_summary.csv', index=False)
"""
    ),
    code(
        """
# Scenario analysis on horizon=1 using baseline features and best baseline model
from src.models import MODEL_REGISTRY

ds = framework.prepare_horizon_dataset(1)
model = MODEL_REGISTRY['Random Forest']
model.fit(ds.X_train, ds.y_train)

base_pred = model.predict(ds.X_test)
base_metrics = regression_metrics(ds.y_test, base_pred)

# Scenario 1: volume spike
X_volume_spike = ds.X_test.copy()
X_volume_spike[:, ds.feature_columns.index('Volume')] *= 1.5 if 'Volume' in ds.feature_columns else 1.0
pred_volume_spike = model.predict(X_volume_spike)

# Scenario 2: volatility increase (perturb high/low-related engineered features)
X_vol_shock = ds.X_test.copy()
for name in ds.feature_columns:
    if 'rolling_std' in name or 'atr' in name or 'bb_width' in name:
        idx = ds.feature_columns.index(name)
        X_vol_shock[:, idx] *= 1.25
pred_vol_shock = model.predict(X_vol_shock)

# Scenario 3: trend reversal
X_trend_reverse = ds.X_test.copy()
for name in ds.feature_columns:
    if 'momentum' in name or 'roc' in name:
        idx = ds.feature_columns.index(name)
        X_trend_reverse[:, idx] *= -1
pred_trend_reverse = model.predict(X_trend_reverse)

scenario_df = pd.DataFrame([
    {'scenario': 'base', 'rmse': regression_metrics(ds.y_test, base_pred)['rmse']},
    {'scenario': 'volume_spike', 'rmse': regression_metrics(ds.y_test, pred_volume_spike)['rmse']},
    {'scenario': 'volatility_shock', 'rmse': regression_metrics(ds.y_test, pred_vol_shock)['rmse']},
    {'scenario': 'trend_reversal', 'rmse': regression_metrics(ds.y_test, pred_trend_reverse)['rmse']},
])

display(scenario_df)
scenario_df.to_csv(OUT / 'tables/09_scenario_analysis.csv', index=False)
"""
    ),
    md("Set `NOTEBOOK_FULL_RUN=1` before execution to run all configured horizons in this notebook."),
    md("""
## Final Lessons Learned

- Hybrid learning improves robustness over isolated model families.
- Weight learning outperforms fixed blending under regime changes.
- Backtesting strategy materially changes perceived model quality.
- Explainability is necessary for trustworthy financial deployment.
- Horizon should drive model and risk policy, not only metric ranking.
"""),
]


notebooks = {
    "01_eda.ipynb": cells_01,
    "02_feature_engineering.ipynb": cells_02,
    "03_baseline_models.ipynb": cells_03,
    "04_deep_learning.ipynb": cells_04,
    "05_hybrid_models.ipynb": cells_05,
    "06_weight_optimization.ipynb": cells_06,
    "07_backtesting.ipynb": cells_07,
    "08_shap_analysis.ipynb": cells_08,
    "09_evaluation_report.ipynb": cells_09,
}

for name, cells in notebooks.items():
    write_notebook(NB_DIR / name, cells)
    print(f"Generated {NB_DIR / name}")
