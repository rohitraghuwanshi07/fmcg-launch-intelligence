"""Analog-launch retrieval (k-NN) and a model-vs-analog ensemble cross-check.

Library: every product whose introduction was observed, represented by (a) attributes known at launch and
(b) its first-26-week demand trajectory per region x channel. For a new product the nearest launches are
found with a ball-tree k-NN over standardised attributes (category match is a hard-ish penalty), and a
kernel-weighted analog forecast is compared with the gradient-boosting forecast. A large disagreement is a
data-driven reliability warning that does not depend on the GBM's own error estimate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from . import config
from .data_ingestion import to_base_size
from .feature_engineering import _pack, ppu100, fit_category_reference

POS_ORD = {p: i for i, p in enumerate(config.POSITIONINGS)}
WEEKS = 26


def build_library(panel: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    d = panel[panel["launch_observed"].astype(bool) & (panel["weeks_since_intro"] < WEEKS)].copy()
    if d.empty:
        return dict(attrs=pd.DataFrame(), traj={}, ref={})
    ref = fit_category_reference(panel)
    d["_bs"] = _pack(d); d["_ppu"] = ppu100(d["price"], d["_bs"])
    attrs = (d.groupby("product_id").agg(category=("category", "first"), brand=("brand", "first"), pack_base=("_bs", "first"),
                                          ppu=("_ppu", "median"), positioning=("positioning", "first"),
                                          mean_units=("units", "mean")).reset_index())
    attrs["price_index"] = attrs.ppu / attrs.category.map(ref)
    attrs["pos_ord"] = attrs.positioning.map(POS_ORD).fillna(1)
    names = catalog.set_index("product_id")["product_name"] if "product_name" in catalog else pd.Series(dtype=str)
    attrs["name"] = attrs.product_id.map(names).fillna(attrs.product_id)
    traj = {pid: g.groupby(["weeks_since_intro", "region", "store_type"])["units"].mean().rename("u").reset_index()
            for pid, g in d.groupby("product_id")}
    return dict(attrs=attrs, traj=traj, ref=ref)


def _query_vector(inp, lib) -> dict:
    base = to_base_size(inp.pack_size, inp.pack_unit)[0]
    ref = lib["ref"].get(inp.category)
    pidx = (inp.price / base * 100) / ref if ref else np.nan
    return dict(category=inp.category, pack_base=base, price_index=pidx, pos_ord=POS_ORD.get(inp.brand_positioning, 1))


def _matrix(attrs: pd.DataFrame, q: dict | None, cat_weight: float = 2.0):
    X = np.column_stack([np.log(attrs.pack_base.clip(lower=1e-6)), np.log(attrs.price_index.clip(lower=1e-6)), attrs.pos_ord])
    return X


def find_analogs(inp, lib: dict, k: int = 5) -> pd.DataFrame:
    attrs = lib["attrs"]
    if attrs.empty:
        return pd.DataFrame()
    q = _query_vector(inp, lib)
    X = _matrix(attrs, q)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    qz = (np.array([[np.log(max(q["pack_base"], 1e-6)), np.log(max(q["price_index"], 1e-6)), q["pos_ord"]]]) - mu) / sd
    same = (attrs.category.values == q["category"]).astype(float)
    CAT_PENALTY = 8.0                                      # different category ranks after every same-category launch (soft: still used if k > category size)
    Zc = np.column_stack([Z, (1 - same) * CAT_PENALTY])
    qc = np.column_stack([qz, [[0.0]]])
    nn = NearestNeighbors(n_neighbors=min(k, len(attrs)), algorithm="ball_tree").fit(Zc)
    dist, idx = nn.kneighbors(qc)
    out = attrs.iloc[idx[0]].copy()
    out["distance"] = dist[0]
    out["same_category"] = same[idx[0]].astype(bool)
    h = max(np.median(dist[0]), 0.5)
    out["weight"] = np.exp(-0.5 * (dist[0] / h) ** 2)
    out["weight"] /= out["weight"].sum()
    return out.reset_index(drop=True)


def analog_trajectory(inp, lib: dict, analogs: pd.DataFrame) -> pd.DataFrame:
    """Per-analog planned-launch trajectory: sum over channels of (mean units per outlet-week x planned outlets)."""
    rows = []
    for _, a in analogs.iterrows():
        t = lib["traj"][a.product_id]
        t = t[t.region.isin(inp.regions)] if inp.regions else t
        g = t.groupby(["weeks_since_intro", "store_type"])["u"].mean().unstack("store_type").reindex(range(WEEKS))
        g = g.ffill().bfill().fillna(0.0)
        tot = sum(g[ch] * n for ch, n in inp.outlets.items() if ch in g and n > 0)
        rows.append(pd.DataFrame(dict(week=np.arange(WEEKS), product_id=a.product_id, name=a["name"], units=np.asarray(tot), weight=a.weight)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def analog_forecast(inp, lib: dict, analogs: pd.DataFrame) -> dict:
    tr = analog_trajectory(inp, lib, analogs)
    if tr.empty:
        return dict(total=np.nan, weekly=pd.DataFrame())
    per = tr.groupby("product_id").apply(lambda g: g.units.sum(), include_groups=False)
    w = analogs.set_index("product_id").weight.reindex(per.index)
    mean_total = float(np.sum(per * w) / w.sum())
    order = np.argsort(per.values)
    cw = np.cumsum(w.values[order]) / w.sum()
    med_total = float(per.values[order][np.searchsorted(cw, 0.5)])
    weekly = tr.assign(wu=tr.units * tr.weight).groupby("week").agg(wu=("wu", "sum"), w=("weight", "sum"))
    weekly = (weekly.wu / weekly.w).rename("units").reset_index()
    return dict(total=mean_total, median_total=med_total, weekly=weekly, per_analog=per, spread=float(np.std(np.log(per.clip(lower=1)))))


def ensemble_check(model_total: float, analog: dict, tol: float = 0.35) -> dict:
    """Compare the GBM forecast with the analog forecast (26-week unit totals, no marketing/price adjustments)."""
    a = analog.get("total", np.nan)
    if not np.isfinite(a) or a <= 0 or model_total <= 0:
        return dict(agreement="n/a", log_gap=np.nan, blended=model_total)
    gap = float(np.log(model_total / a))
    agree = "Strong" if abs(gap) <= tol / 2 else "Moderate" if abs(gap) <= tol else "Weak"
    blended = float(np.exp(0.6 * np.log(model_total) + 0.4 * np.log(a)))   # geometric blend, GBM-weighted
    return dict(agreement=agree, log_gap=gap, gap_pct=100 * (model_total / a - 1), model_total=model_total, analog_total=a,
                blended=blended, tolerance=tol)
