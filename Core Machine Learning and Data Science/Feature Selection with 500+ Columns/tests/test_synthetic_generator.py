import numpy as np

from src.synthetic_generator import compute_informativeness, generate_synthetic_dataset


def test_synthetic_metadata_and_repeated_features_consistent():
    X, y, meta = generate_synthetic_dataset(
        n_samples=200,
        n_features=40,
        n_informative=8,
        n_redundant=4,
        n_repeated=3,
        noise_level=0.05,
        random_state=42,
    )

    assert X.shape == (200, 40)
    assert y.shape[0] == 200

    assert len(meta["informative"]) == 8
    assert len(meta["redundant"]) == 4
    assert len(meta["repeated"]) == 3
    assert len(meta["noise"]) == 25
    assert len(meta["repeated_source_features"]) == 3

    # Repeated columns should be exact copies of their mapped source columns.
    for repeated_col, source_col in zip(meta["repeated"], meta["repeated_source_features"]):
        np.testing.assert_allclose(X[repeated_col].values, X[source_col].values)


def test_compute_informativeness_skips_metadata_helper_keys():
    X, y, meta = generate_synthetic_dataset(
        n_samples=180,
        n_features=30,
        n_informative=6,
        n_redundant=4,
        n_repeated=2,
        random_state=7,
    )

    info = compute_informativeness(X, y, meta)

    assert set(info["feature_type"]) == {"informative", "redundant", "repeated", "noise"}
