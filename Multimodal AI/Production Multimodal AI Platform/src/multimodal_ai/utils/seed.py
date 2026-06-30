"""Deterministic seed setup."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for python and numpy.

    Args:
        seed: Seed value.
    """

    random.seed(seed)
    np.random.seed(seed)
