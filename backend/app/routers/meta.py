from __future__ import annotations
import json
import re
from fastapi import APIRouter, Depends, HTTPException
from src import config, real_data
from ..deps import current_user
from ..services import analytics
from ..services.engine import ENGINE
from ..services.serialize import to_jsonable

router = APIRouter(prefix="/api", tags=["meta"])
STATS = {"requests": 0, "errors": 0}


@router.get("/health")
def health():
    return dict(status="ok" if ENGINE.ready() else "degraded", models_loaded=ENGINE.ctx() is not None, data_loaded=ENGINE.ready())


@router.get("/meta")
def meta():
    return ENGINE.meta()


@router.get("/analytics/overview")
def overview(user=Depends(current_user)):
    if not ENGINE.ready():
        raise HTTPException(503, "No dataset or models.")
    return analytics.overview(ENGINE.panel())


@router.get("/models/metrics")
def model_metrics(user=Depends(current_user)):
    p = config.MODELS_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(503, "Models are not trained.")
    return to_jsonable(json.loads(p.read_text()))


@router.get("/about")
def about():
    root = config.ROOT
    loc = sum(len(p.read_text().splitlines()) for p in (root / "src").glob("*.py"))
    backend_loc = sum(len(p.read_text().splitlines()) for p in (root / "backend").rglob("*.py"))
    tests = sum(len(re.findall(r"^\s*def test_", p.read_text(), re.M)) for p in (root / "tests").glob("*.py"))
    fe = root / "frontend" / "src"
    fe_loc = sum(len(p.read_text().splitlines()) for p in fe.rglob("*.ts*")) if fe.exists() else 0
    out = dict(engine_loc=loc, backend_loc=backend_loc, frontend_loc=fe_loc, test_functions=tests, real_datasets=7 if real_data.available() else 0)
    b = real_data.load_benchmark() if real_data.available() else None
    if b:
        bh = {r["h"]: r for r in b["by_horizon"]}
        out["real_forecast"] = dict(wape_h1=bh[1]["GBM (global)|WAPE"], snaive_h1=bh[1]["Seasonal naive|WAPE"], wape_h12=bh[12]["GBM (global)|WAPE"],
                                    snaive_h12=bh[12]["Seasonal naive|WAPE"], coverage_h3=b["conformal"]["3"]["coverage"], n_series=b["n_series"])
        el = real_data.real_elasticities()["datasets"]
        out["elasticity_range"] = [min(v["elasticity"] for v in el.values()), max(v["elasticity"] for v in el.values())]
    return to_jsonable(out)
