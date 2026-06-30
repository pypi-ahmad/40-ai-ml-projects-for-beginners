import numpy as np

from textclf_framework.evaluation.calibration import expected_calibration_error


def test_ece_within_bounds() -> None:
    probs = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    ece = expected_calibration_error(probs, labels)

    assert 0.0 <= ece <= 1.0
