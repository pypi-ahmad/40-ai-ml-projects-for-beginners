from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from airflow.models import DagBag
from sklearn.linear_model import LinearRegression

from modules.model_registry import load_model, register_model
from modules.settings import get_project_root


def test_model_registry_roundtrip(tmp_path: Path) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = LinearRegression().fit(X, y)

    registered = register_model(
        model=model,
        model_name="unit_test_model",
        base_dir=tmp_path,
        metrics={"rmse": 0.0, "r2": 1.0},
        hyperparameters={"model": "LinearRegression"},
        feature_names=["x"],
        stage="staging",
    )
    assert registered["version"] == 1

    loaded_model, metadata = load_model(base_dir=tmp_path, model_name="unit_test_model", version=1)
    pred = loaded_model.predict(np.array([[4.0]]))[0]
    assert round(float(pred), 6) == 4.0
    assert metadata["model_name"] == "unit_test_model"


def test_dagbag_loads_all_project_dags() -> None:
    root = get_project_root()
    os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow-tests")
    os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

    dagbag = DagBag(dag_folder=str(root / "dags"), include_examples=False)
    assert len(dagbag.import_errors) == 0, dagbag.import_errors

    expected = {
        "01_data_validation",
        "02_feature_engineering",
        "03_model_training",
        "04_reporting",
        "05_end_to_end",
    }
    assert expected.issubset(set(dagbag.dag_ids))
