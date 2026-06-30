import json

import pandas as pd

from src import data_loader


def test_load_isolet_dataset_prefers_local_cache(tmp_path):
    features = pd.DataFrame(
        {
            "f_000": [0.1, 0.2, 0.3],
            "f_001": [1.0, 1.1, 1.2],
            "target": [0, 1, 0],
        }
    )
    meta = {
        "name": "ISOLET",
        "data_id": 44010,
        "n_rows": 3,
        "n_features": 2,
        "n_classes": 2,
    }
    features.to_parquet(tmp_path / "isolet.parquet", index=False)
    (tmp_path / "isolet_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    X, y, metadata = data_loader.load_isolet_dataset(base_dir=tmp_path)

    assert X.shape == (3, 2)
    assert list(X.columns) == ["f_000", "f_001"]
    assert y.tolist() == [0, 1, 0]
    assert metadata["data_id"] == 44010


def test_load_isolet_dataset_downloads_and_caches(monkeypatch, tmp_path):
    frame = pd.DataFrame({"x0": [0.0, 1.0], "x1": [2.0, 3.0]})
    target = pd.Series([1, 2], name="class")

    def fake_fetch_openml(*args, **kwargs):
        return {"data": frame, "target": target}

    monkeypatch.setattr(data_loader, "fetch_openml", fake_fetch_openml)

    X, y, metadata = data_loader.load_isolet_dataset(base_dir=tmp_path, force_refresh=True)

    assert X.shape == (2, 2)
    assert y.tolist() == [1, 2]
    assert metadata["n_features"] == 2
    assert (tmp_path / "isolet.parquet").exists()
    assert (tmp_path / "isolet_metadata.json").exists()
