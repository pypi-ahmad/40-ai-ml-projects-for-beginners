from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.forecast_pipeline import ForecastingFramework



def test_config_has_required_sections(project_root):
    cfg = yaml.safe_load((project_root / "config" / "config.yaml").read_text())
    required = {"data", "features", "models", "backtesting", "weight_optimization", "visualization"}
    assert required.issubset(cfg.keys())



def test_notebook_inventory(project_root):
    notebook_dir = project_root / "notebooks"
    required = [
        "01_eda.ipynb",
        "02_feature_engineering.ipynb",
        "03_baseline_models.ipynb",
        "04_deep_learning.ipynb",
        "05_hybrid_models.ipynb",
        "06_weight_optimization.ipynb",
        "07_backtesting.ipynb",
        "08_shap_analysis.ipynb",
        "09_evaluation_report.ipynb",
    ]
    assert all((notebook_dir / name).exists() for name in required)



def test_streamlit_app_exists(project_root):
    assert (project_root / "app.py").exists()



def test_framework_load_data(project_root):
    fw = ForecastingFramework(str(project_root / "config" / "config.yaml"))
    df = fw.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
