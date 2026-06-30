"""Generate a deterministic curated Ames-style housing dataset snapshot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_api.training.feature_spec import ALL_FEATURES, DEFAULT_CATEGORY_VALUES, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "ames_housing_curated.csv"


def build_snapshot(out_path: Path, n_rows: int = 1460, seed: int = 42) -> pd.DataFrame:
    """Create deterministic Ames-style tabular snapshot and write it to disk."""
    rng = np.random.default_rng(seed)

    frame = pd.DataFrame(
        {
            "lot_frontage": rng.integers(40, 140, size=n_rows),
            "lot_area": rng.integers(2000, 30000, size=n_rows),
            "overall_qual": rng.integers(2, 10, size=n_rows),
            "overall_cond": rng.integers(2, 9, size=n_rows),
            "year_built": rng.integers(1900, 2011, size=n_rows),
            "year_remod_add": rng.integers(1950, 2011, size=n_rows),
            "mas_vnr_area": rng.gamma(2.0, 55.0, size=n_rows).clip(0, 1200),
            "bsmt_fin_sf_1": rng.gamma(2.0, 180.0, size=n_rows).clip(0, 2500),
            "bsmt_fin_sf_2": rng.gamma(1.5, 70.0, size=n_rows).clip(0, 1200),
            "bsmt_unf_sf": rng.gamma(2.2, 120.0, size=n_rows).clip(0, 2200),
            "first_flr_sf": rng.gamma(2.8, 450.0, size=n_rows).clip(400, 3500),
            "second_flr_sf": rng.gamma(2.0, 300.0, size=n_rows).clip(0, 2500),
            "low_qual_fin_sf": rng.gamma(1.1, 35.0, size=n_rows).clip(0, 400),
            "gr_liv_area": rng.gamma(2.9, 650.0, size=n_rows).clip(500, 4500),
            "bsmt_full_bath": rng.integers(0, 3, size=n_rows),
            "bsmt_half_bath": rng.integers(0, 2, size=n_rows),
            "full_bath": rng.integers(1, 4, size=n_rows),
            "half_bath": rng.integers(0, 2, size=n_rows),
            "bedroom_abv_gr": rng.integers(1, 6, size=n_rows),
            "kitchen_abv_gr": rng.integers(1, 3, size=n_rows),
            "tot_rms_abv_grd": rng.integers(3, 13, size=n_rows),
            "fireplaces": rng.integers(0, 3, size=n_rows),
            "garage_yr_blt": rng.integers(1900, 2011, size=n_rows),
            "garage_cars": rng.integers(0, 4, size=n_rows),
            "garage_area": rng.gamma(2.4, 170.0, size=n_rows).clip(0, 1300),
            "wood_deck_sf": rng.gamma(1.7, 60.0, size=n_rows).clip(0, 700),
            "open_porch_sf": rng.gamma(1.8, 45.0, size=n_rows).clip(0, 450),
            "enclosed_porch": rng.gamma(1.4, 30.0, size=n_rows).clip(0, 400),
            "three_ssn_porch": rng.gamma(1.2, 25.0, size=n_rows).clip(0, 350),
            "screen_porch": rng.gamma(1.2, 28.0, size=n_rows).clip(0, 380),
            "pool_area": rng.gamma(1.1, 45.0, size=n_rows).clip(0, 800),
            "misc_val": rng.gamma(1.1, 180.0, size=n_rows).clip(0, 8000),
            "mo_sold": rng.integers(1, 13, size=n_rows),
            "yr_sold": rng.integers(2006, 2011, size=n_rows),
            "ms_subclass": rng.choice([20, 30, 50, 60, 70, 75, 80, 120, 160, 180], size=n_rows),
        }
    )

    frame["total_bsmt_sf"] = frame["bsmt_fin_sf_1"] + frame["bsmt_fin_sf_2"] + frame["bsmt_unf_sf"]

    for feature, values in DEFAULT_CATEGORY_VALUES.items():
        frame[feature] = rng.choice(values, size=n_rows)

    # Keep remod year coherent with build year.
    frame["year_remod_add"] = np.maximum(frame["year_remod_add"], frame["year_built"])
    frame["garage_yr_blt"] = np.maximum(frame["garage_yr_blt"], frame["year_built"] - 2)

    quality_map = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0}
    has_central_air = (frame["central_air"] == "Y").astype(int)

    sale_price = (
        45000
        + frame["overall_qual"] * 32000
        + frame["gr_liv_area"] * 58
        + frame["garage_area"] * 28
        + frame["total_bsmt_sf"] * 20
        + frame["year_built"] * 85
        + frame["neighborhood"].map({"Somerst": 28000, "CollgCr": 18000, "NAmes": 9000, "Edwards": -3000, "OldTown": -7000}).fillna(0)
        + frame["kitchen_qual"].map(quality_map).fillna(2) * 12000
        + has_central_air * 6500
        + rng.normal(0, 18000, size=n_rows)
    )

    frame[TARGET_COLUMN] = np.clip(sale_price, 40000, 800000).round(2)
    frame = frame[ALL_FEATURES + [TARGET_COLUMN]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"Wrote dataset: {out_path}")
    print(f"Rows: {frame.shape[0]}, Columns: {frame.shape[1]}")
    return frame


def main() -> int:
    build_snapshot(OUT_PATH, n_rows=1460, seed=42)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
