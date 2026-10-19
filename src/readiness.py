"""Launch Readiness Index: six transparent 0-100 dimension scores shown on the radar and as a composite."""
from __future__ import annotations
import numpy as np
from . import scenario_engine as se
from .data_ingestion import to_base_size

WEIGHTS = {"Demand & hurdle": 0.30, "Unit economics": 0.20, "Price position": 0.10,
           "Marketing efficiency": 0.10, "Supply readiness": 0.15, "Model confidence": 0.15}


def scores(res: dict, ctx: dict) -> dict:
    inp = se.LaunchInputs.from_dict(res["inputs"])
    ue = res["unit_economics"]["regular"]
    s = {}
    s["Demand & hurdle"] = 100 * float(res["p_hurdle"])
    m = ue["margin_pct"] if np.isfinite(ue["margin_pct"]) else 0.0
    s["Unit economics"] = 100 * float(np.clip(m / (1.5 * max(inp.target_margin, 1e-6)), 0, 1))
    ref = ctx["coldstart"]["ref"].get(inp.category)
    base = to_base_size(inp.pack_size, inp.pack_unit)[0]
    if ref:
        idx = (inp.price / base * 100) / ref
        s["Price position"] = 100 * float(1 - np.clip(abs(np.log(idx)) / 0.5, 0, 1))
    else:
        s["Price position"] = 50.0
    u = res["adjustments"]["marketing_uplift"] / max(inp.max_uplift, 1e-9)
    s["Marketing efficiency"] = 100 * float(np.clip(u, 0, 1))
    inv = res["inventory"]
    s["Supply readiness"] = 100 * float(np.clip(1 - 0.7 * inv["p_stockout"] - 0.3 * inv["p_overstock"], 0, 1))
    lvl = res["reliability"]["level"]
    s["Model confidence"] = {"High": 90.0, "Medium": 65.0}.get(lvl, 35.0)
    return s


def composite(sc: dict) -> float:
    return float(sum(sc[k] * w for k, w in WEIGHTS.items()))


def grade(x: float) -> tuple[str, str]:
    if x >= 80: return "A", "Launch-ready"
    if x >= 65: return "B", "Promising - tighten the weak spots"
    if x >= 50: return "C", "Pilot before committing"
    return "D", "Rework the plan"
