"""Reproducibility utilities for NLP experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable deterministic PyTorch behavior where possible.

    Example:
        >>> set_global_seed(42)
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
