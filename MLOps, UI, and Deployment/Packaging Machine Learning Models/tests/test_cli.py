import argparse
import json
from pathlib import Path

import pytest

from ml_package.cli import predict as cli_predict
from ml_package.settings import PackageSettings


def _settings(tmp_path: Path) -> PackageSettings:
    return PackageSettings(
        model_path=tmp_path / "iris_model.pkl",
        registry_path=tmp_path / "registry.json",
        background_data_path=tmp_path / "background_data.npy",
        log_dir=tmp_path / "logs",
        cors_origins=["http://localhost"],
        verify_artifacts=True,
        allow_unsafe_deserialization=False,
        trusted_digests=set(),
        api_host="0.0.0.0",
        api_port=8000,
    )


def test_load_samples_from_json_object(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ),
        encoding="utf-8",
    )

    rows = cli_predict.load_samples_from_json(path)
    assert len(rows) == 1
    assert rows[0]["sepal_length"] == 5.1


def test_load_samples_from_json_missing_field(tmp_path):
    path = tmp_path / "bad_samples.json"
    path.write_text(json.dumps([{"sepal_length": 5.1}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        cli_predict.load_samples_from_json(path)


def test_load_samples_from_csv_valid(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "sepal_length,sepal_width,petal_length,petal_width\n"
        "5.1,3.5,1.4,0.2\n"
        "6.0,2.9,4.5,1.5\n",
        encoding="utf-8",
    )

    rows = cli_predict.load_samples_from_csv(path)
    assert len(rows) == 2
    assert rows[1]["petal_width"] == 1.5


def test_load_samples_from_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "sepal_length,sepal_width,petal_length,petal_width\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        cli_predict.load_samples_from_csv(path)


def test_load_batch_samples_rejects_unknown_extension(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json or \.csv"):
        cli_predict.load_batch_samples(path)


def test_cmd_predict_success(monkeypatch, tmp_path):
    class DummyEngine:
        def predict(self, features):
            assert features.shape == (1, 4)
            return {
                "prediction": 0,
                "species": "setosa",
                "confidence": 0.99,
                "probabilities": [0.99, 0.005, 0.005],
                "latency_ms": 1.0,
                "model_name": "iris_classifier",
                "model_version": "v2",
            }

    emitted = {}

    monkeypatch.setattr(cli_predict, "load_model", lambda *args, **kwargs: DummyEngine())
    monkeypatch.setattr(cli_predict, "emit_payload", lambda payload, output: emitted.update(payload))

    args = argparse.Namespace(
        model_path=str(tmp_path / "model.pkl"),
        output=None,
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2,
    )
    code = cli_predict.cmd_predict(args, _settings(tmp_path))
    assert code == 0
    assert emitted["species"] == "setosa"


def test_main_returns_error_code_for_bad_command(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        cli_predict.sys,
        "argv",
        ["ml-predict", "--model-path", str(tmp_path / "missing.pkl"), "batch", "missing.json"],
    )

    code = cli_predict.main()
    captured = capsys.readouterr()

    assert code == 2
    assert "command_failed" in captured.out
