"""Reproducibility seed helpers."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Set process-level deterministic seeds where possible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass
