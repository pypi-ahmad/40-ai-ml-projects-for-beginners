import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml_package.artifact_security import compute_sha256, manifest_path_for
from ml_package.exceptions import ArtifactVerificationError, UnsafeDeserializationError
from ml_package.model_loader import ModelLoader


@pytest.fixture
def trained_model():
    X = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5],
        ]
    )
    y = np.array([0, 1, 2])
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model


def test_save_creates_manifest(trained_model, tmp_path):
    path = tmp_path / "secure_model.pkl"
    loader = ModelLoader(path)
    loader.save(trained_model)

    assert path.exists()
    assert manifest_path_for(path).exists()


def test_require_manifest_blocks_legacy_artifact(trained_model, tmp_path):
    path = tmp_path / "legacy_model.pkl"
    with path.open("wb") as handle:
        pickle.dump(trained_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loader = ModelLoader(path, verify_integrity=True, require_manifest=True)
    with pytest.raises(ArtifactVerificationError, match="Missing manifest"):
        loader.load()


def test_allow_list_digest_enforced(trained_model, tmp_path):
    path = tmp_path / "allowlist_model.pkl"
    save_loader = ModelLoader(path)
    save_loader.save(trained_model)

    trusted_digest = compute_sha256(path)
    allowed_loader = ModelLoader(path, trusted_digests={trusted_digest})
    allowed_loader.load()

    blocked_loader = ModelLoader(path, trusted_digests={"bad-digest"})
    with pytest.raises(ArtifactVerificationError, match="allow-list"):
        blocked_loader.load()


def test_unsafe_deserialization_can_be_blocked(trained_model, tmp_path):
    path = tmp_path / "unsafe_model.pkl"
    save_loader = ModelLoader(path)
    save_loader.save(trained_model)

    blocked_loader = ModelLoader(path, allow_unsafe_deserialization=False)
    with pytest.raises(UnsafeDeserializationError, match="Unsafe deserialization blocked"):
        blocked_loader.load()


def test_unsafe_deserialization_allowed_when_digest_trusted(trained_model, tmp_path):
    path = tmp_path / "unsafe_trusted_model.pkl"
    save_loader = ModelLoader(path)
    save_loader.save(trained_model)

    trusted_digest = compute_sha256(path)
    trusted_loader = ModelLoader(
        path,
        allow_unsafe_deserialization=False,
        trusted_digests={trusted_digest},
        verify_integrity=True,
        require_manifest=True,
    )

    loaded = trusted_loader.load()
    prediction = loaded.predict([[5.1, 3.5, 1.4, 0.2]])
    assert int(prediction[0]) in [0, 1, 2]
