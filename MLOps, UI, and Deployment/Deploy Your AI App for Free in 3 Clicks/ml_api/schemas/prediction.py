"""Pydantic request schemas for prediction and explainability endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HouseFeatures(BaseModel):
    """Full raw feature contract for curated Ames Housing inference."""

    model_config = ConfigDict(extra="forbid")

    lot_frontage: float = Field(ge=20, le=350)
    lot_area: int = Field(ge=1000, le=300000)
    overall_qual: int = Field(ge=1, le=10)
    overall_cond: int = Field(ge=1, le=10)
    year_built: int = Field(ge=1870, le=2030)
    year_remod_add: int = Field(ge=1950, le=2030)
    mas_vnr_area: float = Field(ge=0, le=2000)
    bsmt_fin_sf_1: float = Field(ge=0, le=3000)
    bsmt_fin_sf_2: float = Field(ge=0, le=2000)
    bsmt_unf_sf: float = Field(ge=0, le=3000)
    total_bsmt_sf: float = Field(ge=0, le=5000)
    first_flr_sf: float = Field(ge=300, le=5000)
    second_flr_sf: float = Field(ge=0, le=3000)
    low_qual_fin_sf: float = Field(ge=0, le=1000)
    gr_liv_area: float = Field(ge=300, le=8000)
    bsmt_full_bath: int = Field(ge=0, le=4)
    bsmt_half_bath: int = Field(ge=0, le=2)
    full_bath: int = Field(ge=0, le=4)
    half_bath: int = Field(ge=0, le=3)
    bedroom_abv_gr: int = Field(ge=0, le=10)
    kitchen_abv_gr: int = Field(ge=0, le=4)
    tot_rms_abv_grd: int = Field(ge=1, le=20)
    fireplaces: int = Field(ge=0, le=5)
    garage_yr_blt: int = Field(ge=1890, le=2030)
    garage_cars: int = Field(ge=0, le=6)
    garage_area: float = Field(ge=0, le=2000)
    wood_deck_sf: float = Field(ge=0, le=1000)
    open_porch_sf: float = Field(ge=0, le=600)
    enclosed_porch: float = Field(ge=0, le=800)
    three_ssn_porch: float = Field(ge=0, le=600)
    screen_porch: float = Field(ge=0, le=600)
    pool_area: float = Field(ge=0, le=1500)
    misc_val: float = Field(ge=0, le=20000)
    mo_sold: int = Field(ge=1, le=12)
    yr_sold: int = Field(ge=2006, le=2030)
    ms_subclass: int = Field(ge=20, le=190)

    ms_zoning: str = Field(min_length=1, max_length=20)
    street: str = Field(min_length=1, max_length=20)
    lot_shape: str = Field(min_length=1, max_length=20)
    land_contour: str = Field(min_length=1, max_length=20)
    utilities: str = Field(min_length=1, max_length=20)
    lot_config: str = Field(min_length=1, max_length=20)
    land_slope: str = Field(min_length=1, max_length=20)
    neighborhood: str = Field(min_length=1, max_length=40)
    condition_1: str = Field(min_length=1, max_length=20)
    bldg_type: str = Field(min_length=1, max_length=20)
    house_style: str = Field(min_length=1, max_length=20)
    roof_style: str = Field(min_length=1, max_length=20)
    exterior_1st: str = Field(min_length=1, max_length=30)
    exterior_2nd: str = Field(min_length=1, max_length=30)
    exter_qual: str = Field(min_length=1, max_length=10)
    exter_cond: str = Field(min_length=1, max_length=10)
    foundation: str = Field(min_length=1, max_length=20)
    bsmt_qual: str = Field(min_length=1, max_length=10)
    bsmt_cond: str = Field(min_length=1, max_length=10)
    bsmt_exposure: str = Field(min_length=1, max_length=10)
    bsmt_fin_type_1: str = Field(min_length=1, max_length=10)
    bsmt_fin_type_2: str = Field(min_length=1, max_length=10)
    heating: str = Field(min_length=1, max_length=20)
    heating_qc: str = Field(min_length=1, max_length=10)
    central_air: Literal["Y", "N"]
    electrical: str = Field(min_length=1, max_length=20)
    kitchen_qual: str = Field(min_length=1, max_length=10)
    functional: str = Field(min_length=1, max_length=20)
    fireplace_qu: str = Field(min_length=1, max_length=10)
    garage_type: str = Field(min_length=1, max_length=20)
    garage_finish: str = Field(min_length=1, max_length=10)
    garage_qual: str = Field(min_length=1, max_length=10)
    garage_cond: str = Field(min_length=1, max_length=10)
    paved_drive: Literal["Y", "P", "N"]
    sale_type: str = Field(min_length=1, max_length=20)
    sale_condition: str = Field(min_length=1, max_length=20)

    @field_validator("year_remod_add")
    @classmethod
    def remod_not_before_built(cls, value: int, info):
        year_built = info.data.get("year_built")
        if year_built is not None and value < year_built:
            raise ValueError("year_remod_add must be >= year_built")
        return value

    @field_validator("garage_yr_blt")
    @classmethod
    def garage_year_sane(cls, value: int, info):
        year_built = info.data.get("year_built")
        if year_built is not None and value < (year_built - 5):
            raise ValueError("garage_yr_blt appears inconsistent with year_built")
        return value


class PredictionRequest(BaseModel):
    record: HouseFeatures


class BatchPredictionRequest(BaseModel):
    records: list[HouseFeatures] = Field(min_length=1, max_length=512)


class ExplainRequest(BaseModel):
    record: HouseFeatures
    top_k: int = Field(default=10, ge=1, le=40)
