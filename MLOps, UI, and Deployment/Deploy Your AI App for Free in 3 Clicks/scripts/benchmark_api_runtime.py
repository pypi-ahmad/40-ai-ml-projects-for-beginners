"""Benchmark API route execution latency without socket-bound clients."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

from ml_api.api.routes import metrics, predict, predict_batch
from ml_api.app import create_app
from ml_api.core.config import get_settings
from ml_api.schemas.prediction import BatchPredictionRequest, HouseFeatures, PredictionRequest

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "outputs" / "metrics" / "fastapi_runtime_benchmark.json"


def _sample_record() -> HouseFeatures:
    return HouseFeatures(
        lot_frontage=80,
        lot_area=9600,
        overall_qual=7,
        overall_cond=5,
        year_built=2003,
        year_remod_add=2005,
        mas_vnr_area=180.0,
        bsmt_fin_sf_1=650.0,
        bsmt_fin_sf_2=0.0,
        bsmt_unf_sf=330.0,
        total_bsmt_sf=980.0,
        first_flr_sf=1030.0,
        second_flr_sf=820.0,
        low_qual_fin_sf=0.0,
        gr_liv_area=1850.0,
        bsmt_full_bath=1,
        bsmt_half_bath=0,
        full_bath=2,
        half_bath=1,
        bedroom_abv_gr=3,
        kitchen_abv_gr=1,
        tot_rms_abv_grd=8,
        fireplaces=1,
        garage_yr_blt=2003,
        garage_cars=2,
        garage_area=520.0,
        wood_deck_sf=120.0,
        open_porch_sf=55.0,
        enclosed_porch=0.0,
        three_ssn_porch=0.0,
        screen_porch=0.0,
        pool_area=0.0,
        misc_val=0.0,
        mo_sold=6,
        yr_sold=2008,
        ms_subclass=60,
        ms_zoning="RL",
        street="Pave",
        lot_shape="Reg",
        land_contour="Lvl",
        utilities="AllPub",
        lot_config="Inside",
        land_slope="Gtl",
        neighborhood="CollgCr",
        condition_1="Norm",
        bldg_type="1Fam",
        house_style="2Story",
        roof_style="Gable",
        exterior_1st="VinylSd",
        exterior_2nd="VinylSd",
        exter_qual="Gd",
        exter_cond="TA",
        foundation="PConc",
        bsmt_qual="Gd",
        bsmt_cond="TA",
        bsmt_exposure="Av",
        bsmt_fin_type_1="GLQ",
        bsmt_fin_type_2="Unf",
        heating="GasA",
        heating_qc="Ex",
        central_air="Y",
        electrical="SBrkr",
        kitchen_qual="Gd",
        functional="Typ",
        fireplace_qu="Gd",
        garage_type="Attchd",
        garage_finish="RFn",
        garage_qual="TA",
        garage_cond="TA",
        paved_drive="Y",
        sale_type="WD",
        sale_condition="Normal",
    )


def main() -> int:
    settings = get_settings()

    startup_t0 = time.perf_counter()
    app = create_app()
    app.state.model_service.load()
    startup_ms = (time.perf_counter() - startup_t0) * 1000

    request = SimpleNamespace(app=app, state=SimpleNamespace(request_id="benchmark-request"))

    # Warm-up
    predict(PredictionRequest(record=_sample_record()), request=request, model_service=app.state.model_service)

    t0 = time.perf_counter()
    single = predict(PredictionRequest(record=_sample_record()), request=request, model_service=app.state.model_service)
    single_ms = (time.perf_counter() - t0) * 1000
    app.state.metrics_store.record("/predict", 200, single_ms)

    batch_size = 32
    batch_payload = BatchPredictionRequest(records=[_sample_record() for _ in range(batch_size)])
    t1 = time.perf_counter()
    batch = predict_batch(
        payload=batch_payload,
        request=request,
        settings=settings,
        model_service=app.state.model_service,
    )
    batch_ms = (time.perf_counter() - t1) * 1000
    app.state.metrics_store.record("/predict-batch", 200, batch_ms)

    metrics_snapshot = metrics(metrics_store=app.state.metrics_store, model_service=app.state.model_service)

    payload = {
        "startup_ms": round(startup_ms, 3),
        "single_predict_ms": round(single_ms, 3),
        "batch_predict_ms": round(batch_ms, 3),
        "batch_size": batch_size,
        "batch_rows_per_second": round(batch_size / max(batch_ms / 1000, 1e-6), 3),
        "single_prediction": single.model_dump(),
        "batch_prediction": batch.model_dump(),
        "metrics_snapshot": metrics_snapshot.model_dump(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
