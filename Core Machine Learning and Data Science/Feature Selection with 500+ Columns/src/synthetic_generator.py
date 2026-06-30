"""
synthetic_generator.py
-----------------------
Generates synthetic classification datasets with many features (500+).

Purpose:
  Demonstrate feature selection concepts in a controlled environment where
  we know exactly which features are informative, redundant, and noise.

This mirrors the classic blog-example approach but is more configurable
and production-quality.

Inputs:
  - n_samples: number of rows
  - n_features: total number of features (default 500)
  - n_informative: how many features actually predict the target
  - n_redundant: linear combinations of informative features
  - n_repeated: duplicated features
  - noise_level: standard deviation of noise added to features
  - random_state: reproducibility seed

Outputs:
  - X: feature matrix (n_samples x n_features)
  - y: binary target vector (0/1)
  - feature_metadata: dict mapping feature indices to type
    ('informative', 'redundant', 'repeated', 'noise')
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 500,
    n_informative: int = 20,
    n_redundant: int = 10,
    n_repeated: int = 5,
    n_clusters_per_class: int = 2,
    noise_level: float = 0.1,
    flip_y: float = 0.01,
    random_state: int = 42,
    feature_prefix: str = "feat",
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Generate a synthetic classification dataset with known feature structure.

    How it works:
      1. sklearn's make_classification creates the core dataset with
         informative + redundant + repeated features.
      2. We add extra pure-noise features to reach n_features.
      3. We track which features are which in a metadata dict.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows).
    n_features : int
        Total number of features (columns).
    n_informative : int
        Number of features that directly carry signal.
    n_redundant : int
        Number of features that are linear combinations of informative ones.
    n_repeated : int
        Number of features that are exact copies of others.
    n_clusters_per_class : int
        Clusters per class (more = harder classification boundary).
    noise_level : float
        Standard deviation of Gaussian noise added.
    flip_y : float
        Fraction of labels randomly flipped to make the task less trivial.
    random_state : int
        Seed for reproducibility.
    feature_prefix : str
        Prefix for column names.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix. Shape (n_samples, n_features).
    y : pd.Series
        Binary target (0 or 1).
    feature_metadata : dict
        Maps feature type to list of column names:
        {
            'informative': [...],
            'redundant': [...],
            'repeated': [...],
            'noise': [...]
        }
    """
    rng = np.random.RandomState(random_state)

    # --- Stage 1: Core informative + redundant features ---
    n_base = n_informative + n_redundant
    n_core = n_base + n_repeated

    if n_base <= 0:
        raise ValueError("n_informative + n_redundant must be > 0")

    n_noise = n_features - n_core
    if n_noise < 0:
        raise ValueError(
            f"n_features ({n_features}) must be >= "
            f"n_informative + n_redundant + n_repeated "
            f"({n_core})"
        )

    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_base,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_clusters_per_class=n_clusters_per_class,
        flip_y=flip_y,
        random_state=random_state,
        shuffle=False,
    )

    # Add Gaussian noise to base columns first
    if noise_level > 0:
        X_base = X_base + (rng.randn(*X_base.shape) * noise_level)

    # --- Stage 2: Create repeated columns as exact copies of base columns ---
    repeated_source_idx: np.ndarray
    if n_repeated > 0:
        repeated_source_idx = rng.choice(n_base, size=n_repeated, replace=True)
        X_repeated = X_base[:, repeated_source_idx].copy()
    else:
        repeated_source_idx = np.array([], dtype=int)
        X_repeated = np.empty((n_samples, 0))

    # --- Stage 3: Add pure noise features ---
    X_noise = rng.randn(n_samples, n_noise)

    # --- Stage 4: Combine ---
    X_combined = np.hstack([X_base, X_repeated, X_noise])

    # --- Create feature metadata ---
    feat_names = [f"{feature_prefix}_{i:04d}" for i in range(n_features)]
    X_df = pd.DataFrame(X_combined, columns=feat_names)
    y_series = pd.Series(y, name="target")

    informative_names = feat_names[:n_informative]
    redundant_names = feat_names[n_informative:n_base]
    repeated_names = feat_names[n_base:n_core]
    noise_names = feat_names[n_core:]

    feature_metadata = {
        "informative": informative_names,
        "redundant": redundant_names,
        "repeated": repeated_names,
        "noise": noise_names,
        "repeated_source_features": [feat_names[idx] for idx in repeated_source_idx],
    }

    return X_df, y_series, feature_metadata


def compute_informativeness(
    X: pd.DataFrame, y: pd.Series, feature_metadata: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Compute how well each feature type separates the classes.

    Uses ANOVA F-statistic between informative/redundant/repeated/noise
    groups to verify the generated data matches expectations.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target.
    feature_metadata : dict
        Feature type mapping from generate_synthetic_dataset().

    Returns
    -------
    pd.DataFrame
        Summary with columns: [feature_type, count, mean_f_stat, std_f_stat]
    """
    from sklearn.feature_selection import f_classif

    results = []
    valid_groups = {"informative", "redundant", "repeated", "noise"}
    for ftype, feats in feature_metadata.items():
        if ftype not in valid_groups:
            continue
        if not feats:
            continue
        f_stats, _ = f_classif(X[feats], y)
        results.append({
            "feature_type": ftype,
            "count": len(feats),
            "mean_f_stat": float(np.mean(f_stats)),
            "std_f_stat": float(np.std(f_stats)),
        })

    return pd.DataFrame(results)


def generate_noise_scale_experiment(
    noise_levels: List[float] = [0.0, 0.1, 0.5, 1.0, 2.0],
    n_samples: int = 500,
    n_features: int = 100,
    n_informative: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate multiple datasets at different noise levels and return
    the ANOVA F-statistics to show how noise buries signal.

    This is useful for teaching: as noise increases, informative features
    become indistinguishable from noise features.

    Parameters
    ----------
    noise_levels : list of float
        Noise standard deviations to test.
    n_samples, n_features, n_informative : int
        Dataset parameters.
    random_state : int
        Seed.

    Returns
    -------
    pd.DataFrame
        Columns: [noise_level, feature_type, mean_f_stat]
    """
    rows = []
    for noise in noise_levels:
        X, y, meta = generate_synthetic_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=5,
            n_repeated=2,
            noise_level=noise,
            random_state=random_state,
        )
        info_df = compute_informativeness(X, y, meta)
        for _, r in info_df.iterrows():
            rows.append({
                "noise_level": noise,
                "feature_type": r["feature_type"],
                "mean_f_stat": r["mean_f_stat"],
            })
    return pd.DataFrame(rows)
