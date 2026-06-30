"""
Utility helpers for the Geospatial Clustering project.

Provides small, reusable functions for:
- Seeding random number generators for reproducibility.
- Directory creation.
- Timing function execution.
- Loading configuration from YAML / JSON files.
- Safe DataFrame writing.
- Converting lat/lon pairs to GeoDataFrames (optional geopandas).

These are building blocks used by the pipeline and individual modules.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import (
    INDIA_BOUNDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Set all common random seeds for reproducible results.

    Seeds ``random``, ``numpy``, and (if available) ``torch`` and
    ``tensorflow``.  Always call this before any stochastic operation.

    Parameters
    ----------
    seed : int
        Seed value (default 42, the answer to everything).

    Example
    -------
    >>> from src.utils import set_seed
    >>> set_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: F811
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf  # noqa: F811
        tf.random.set_seed(seed)
    except ImportError:
        pass

    logger.debug("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path.  Can be a file path — only the parent is created.

    Returns
    -------
    str
        The input ``path`` unchanged, for chaining.

    Example
    -------
    >>> filepath = ensure_dir("outputs/figures/my_plot.png")
    >>> # outputs/figures/ now exists; my_plot.png is the filename part
    """
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path
    os.makedirs(directory, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------


def timer_decorator(func: Callable) -> Callable:
    """
    Decorator that logs the execution duration of a function.

    Usage
    -----
    >>> @timer_decorator
    ... def my_function():
    ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            "%s executed in %.4f seconds",
            func.__name__,
            elapsed,
        )
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Supports ``.yaml``, ``.yml``, and ``.json`` extensions.

    Parameters
    ----------
    path : str
        Path to the config file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is not recognised.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # noqa: F811
            except ImportError:
                raise ImportError("PyYAML is required for .yaml files: pip install pyyaml")
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}. Use .yaml, .yml, or .json.")


# ---------------------------------------------------------------------------
# DataFrame I/O
# ---------------------------------------------------------------------------


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    format: str = "csv",
    **kwargs: Any,
) -> str:
    """
    Save a DataFrame to disk with error handling.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : str
        Destination path.
    format : str
        ``"csv"`` (default), ``"parquet"``, or ``"pkl"`` (pickle).
    **kwargs
        Additional arguments passed to the underlying writer (e.g. ``index=False``).

    Returns
    -------
    str
        The path the file was saved to.

    Example
    -------
    >>> save_dataframe(df, "outputs/data.csv", index=False)
    """
    path = ensure_dir(path)
    kwargs.setdefault("index", False)

    try:
        if format == "csv":
            df.to_csv(path, **kwargs)
        elif format == "parquet":
            df.to_parquet(path, **kwargs)
        elif format == "pkl":
            df.to_pickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Saved %d rows → %s", len(df), path)
    except Exception as e:
        logger.error("Failed to save DataFrame to %s: %s", path, e)
        raise

    return path


# ---------------------------------------------------------------------------
# GeoPandas conversion
# ---------------------------------------------------------------------------


def coords_to_gdf(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
) -> Any:
    """
    Convert a DataFrame with lat/lon columns to a GeoDataFrame.

    Requires ``geopandas`` and ``shapely``.  If they are not installed,
    raises an ``ImportError`` with a helpful message.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    crs : str
        Coordinate reference system string (default EPSG:4326 = WGS84).

    Returns
    -------
    gpd.GeoDataFrame
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError(
            "geopandas and shapely are required: pip install geopandas shapely"
        ) from e

    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf


# ---------------------------------------------------------------------------
# Coordinate validation helpers
# ---------------------------------------------------------------------------


def is_within_india(lat: float, lon: float) -> bool:
    """
    Check whether a (lat, lon) pair falls within approximate India bounds.

    Parameters
    ----------
    lat : float
    lon : float

    Returns
    -------
    bool
    """
    return (
        INDIA_BOUNDS["lat_min"] <= lat <= INDIA_BOUNDS["lat_max"]
        and INDIA_BOUNDS["lon_min"] <= lon <= INDIA_BOUNDS["lon_max"]
    )


def is_valid_coordinate(lat: float, lon: float) -> bool:
    """
    Basic sanity check for a geographic coordinate.

    Returns ``True`` if lat ∈ [-90, 90], lon ∈ [-180, 180], and
    neither is zero (zero coordinates are a common data-entry error).

    Parameters
    ----------
    lat : float
    lon : float

    Returns
    -------
    bool
    """
    valid_range = (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)
    non_zero = (lat != 0.0) or (lon != 0.0)
    return valid_range and non_zero
