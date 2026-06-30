import json
from pathlib import Path

from ml_package.settings import PackageSettings


def test_allow_unsafe_deserialization_default_is_false(monkeypatch):
    monkeypatch.delenv("ML_ALLOW_UNSAFE_DESERIALIZATION", raising=False)
    settings = PackageSettings.from_env()
    assert settings.allow_unsafe_deserialization is False


def test_resolved_trusted_digests_reads_active_and_matching_version(tmp_path):
    registry_path = tmp_path / "registry.json"
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"placeholder")

    payload = {
        "schema_version": 1,
        "active_version": "v2",
        "versions": {
            "v1": {
                "version_id": "v1",
                "model_path": str(tmp_path / "old.pkl"),
                "artifact_sha256": "digest-v1",
            },
            "v2": {
                "version_id": "v2",
                "model_path": str(model_path),
                "artifact_sha256": "digest-v2",
            },
        },
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    settings = PackageSettings(
        model_path=model_path,
        registry_path=registry_path,
        background_data_path=tmp_path / "background.npy",
        log_dir=tmp_path / "logs",
        cors_origins=["http://localhost"],
        verify_artifacts=True,
        allow_unsafe_deserialization=False,
        trusted_digests={"digest-manual"},
        api_host="0.0.0.0",
        api_port=8000,
    )
    digests = settings.resolved_trusted_digests(model_path)
    assert digests == {"digest-manual", "digest-v2"}

