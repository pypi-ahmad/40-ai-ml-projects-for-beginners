from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds for python, numpy, and torch if installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch is optional for utilities.
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_path(path: str | Path) -> Path:
    """Return Path object for str/pathlib inputs."""
    return path if isinstance(path, Path) else Path(path)
