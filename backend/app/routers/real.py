from __future__ import annotations
import json
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from src import config, real_data as rd
from ..deps import current_user
from ..services.serialize import to_jsonable

router = APIRouter(prefix="/api/real", tags=["real-data"], dependencies=[Depends(current_user)])


def _need():
    if not rd.available():
        raise HTTPException(503, "Real datasets are not bundled (data/real).")


@router.get("/elasticities")
def elasticities():
    _need()
    out = rd.real_elasticities()
    sim = {}
    pj = config.MODELS_DIR / "pricing.json"
    if pj.exists(): sim = json.loads(pj.read_text()).get("elasticity", {})
    return to_jsonable(dict(out, simulated=sim))


@router.get("/benchmark")
def benchmark():
    _need(); b = rd.load_benchmark()
    if b is None: raise HTTPException(503, "Run `python scripts/train_real.py` to build the real-data benchmark.")
    return to_jsonable(b)


@router.get("/industries")
def industries():
    _need(); r = rd.load_retail()
    return sorted(r.industry.unique().tolist())


@router.get("/seasonality")
def seasonality(industry: str):
    _need(); r = rd.load_retail()
    if industry not in set(r.industry): raise HTTPException(404, "Unknown industry.")
    return to_jsonable(rd.seasonal_profile(industry))


@router.get("/series")
def series():
    _need(); r = rd.load_retail().drop_duplicates("sid")
    return to_jsonable(r.assign(label=r.industry + " | " + r.state)[["sid", "label", "industry", "state"]])


@router.get("/series/{sid}")
def series_forecast(sid: str):
    _need(); r = rd.load_retail(); h = r[r.sid == sid].tail(72)
    if h.empty: raise HTTPException(404, "Unknown series.")
    f = pd.read_csv(config.DATA_REAL / "derived" / "future_forecast.csv.gz", parse_dates=["target_date"])
    f = f[f.sid == sid].sort_values("h")
    return to_jsonable(dict(history=h[["date", "y"]], forecast=f[["h", "target_date", "forecast", "lo", "hi"]]))


@router.get("/provenance")
def provenance():
    _need(); return json.loads((config.DATA_REAL / "PROVENANCE.json").read_text())
