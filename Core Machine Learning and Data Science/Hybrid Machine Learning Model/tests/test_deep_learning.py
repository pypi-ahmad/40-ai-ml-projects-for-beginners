from __future__ import annotations

import numpy as np

from src.deep_learning import scale_for_sequences


def test_scale_for_sequences_fit_on_train_only():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    X_val = np.array([[2.5], [3.5]])
    X_test = np.array([[10.0], [12.0]])
    y_train = np.array([1.0, 1.5, 2.0, 2.5])
    y_val = np.array([2.2, 2.7])
    y_test = np.array([5.0, 6.0])

    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, _, _ = scale_for_sequences(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Values beyond train max should remain >1 when scaler is fit only on train split.
    assert X_test_s.max() > 1.0
    assert y_test_s.max() > 1.0
    assert X_train_s.min() >= 0.0 and X_train_s.max() <= 1.0
    assert y_train_s.min() >= 0.0 and y_train_s.max() <= 1.0
