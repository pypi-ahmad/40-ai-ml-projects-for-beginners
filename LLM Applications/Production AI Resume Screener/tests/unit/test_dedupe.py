import numpy as np

from resume_ai.ingestion.dedupe import find_near_duplicate


def test_find_near_duplicate() -> None:
    current = np.array([1.0, 0.0])
    existing = [
        (1, np.array([0.99, 0.01])),
        (2, np.array([0.1, 0.9])),
    ]
    match = find_near_duplicate(current_embedding=current, existing_embeddings=existing, threshold=0.95)
    assert match == 1
