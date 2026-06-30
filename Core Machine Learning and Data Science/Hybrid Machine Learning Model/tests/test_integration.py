from __future__ import annotations

import pandas as pd

from src.forecast_pipeline import ForecastingFramework
from src.models import train_model



def test_framework_prepare_horizon_dataset(project_root):
    fw = ForecastingFramework(str(project_root / "config" / "config.yaml"))
    fw.load_data()
    ds = fw.prepare_horizon_dataset(1)
    assert ds.X_train.shape[0] > 0
    assert ds.X_val.shape[0] > 0
    assert ds.X_test.shape[0] > 0
    assert len(ds.feature_columns) == ds.X_train.shape[1]



def test_framework_fast_training_step(project_root):
    fw = ForecastingFramework(str(project_root / "config" / "config.yaml"))
    fw.load_data()
    ds = fw.prepare_horizon_dataset(1)
    model, metrics = train_model(ds.X_train, ds.y_train, model_name="Linear Regression")
    pred = model.predict(ds.X_test)
    assert len(pred) == len(ds.y_test)
    assert metrics["train_rmse"] >= 0
