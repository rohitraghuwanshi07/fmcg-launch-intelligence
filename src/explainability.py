"""Explainability and reliability. Importance != causation: all outputs are labelled as model sensitivity."""
from __future__ import annotations
import numpy as np
import pandas as pd

def assess_reliability(inp, ctx) -> dict:
    cs = ctx["coldstart"]; sup = cs["support"]
    flags, score = [], 100
    n_cat = sup["products_per_category"].get(inp.category, 0)
    if n_cat == 0:
        flags.append(("high", f"Category '{inp.category}' has no launch history in training data.")); score -= 50
    elif n_cat < 6:
        flags.append(("high", f"Only {n_cat} observed launches in this category.")); score -= 25
    elif n_cat < 10:
        flags.append(("medium", f"Only {n_cat} observed launches in this category.")); score -= 10
    from .data_ingestion import to_base_size
    base = to_base_size(inp.pack_size, inp.pack_unit)[0]
    ref = cs["ref"].get(inp.category)
    if ref:
        idx = (inp.price / base * 100) / ref
        lo, hi = sup["price_index_range"]
        if not lo <= idx <= hi:
            flags.append(("high", f"Price index {idx:.2f} (vs category median) is outside the supported range {lo:.2f}-{hi:.2f}: extrapolation.")); score -= 20
    lo, hi = sup["pack_size_range"]
    if not lo <= base <= hi:
        flags.append(("medium", f"Pack size {base:g} is outside the training range {lo:g}-{hi:g}.")); score -= 10
    if inp.intro_discount * 100 > sup["discount_range"][1]:
        flags.append(("medium", "Introductory discount is deeper than any discount observed in training data.")); score -= 10
    t, b = cs["metrics"]["test"], None
    sel = next((k for k in t if "[selected]" in k), None)
    if sel and t["Baseline"]["WAPE"] <= t[sel]["WAPE"]:
        flags.append(("medium", "On held-out products the model did not beat the simple category baseline; treat point forecasts cautiously.")); score -= 15
    if cs["n_ratio_products"] < 30:
        flags.append(("medium", f"Uncertainty is calibrated on only {cs['n_ratio_products']} launches.")); score -= 10
    sd = float(np.std(cs["log_ratio"]))
    if sd > 0.5:
        flags.append(("medium", f"Out-of-sample launch totals differ from forecasts by a typical factor of exp({sd:.2f}) = x{np.exp(sd):.2f}.")); score -= 10
    if inp.hurdle_weeks > 26 or any(w > 26 for w in config_horizon_weeks()):
        flags.append(("low", "Uncertainty was calibrated on 26-week totals; 12-month ranges are likely understated.")); score -= 0
    if ctx.get("demo", True):
        flags.append(("high", "DEMO MODE: models were trained on SIMULATED data; outputs illustrate the workflow only.")); score = min(score, 40)
    level = "High" if score >= 75 else "Moderate" if score >= 50 else "Low"
    return dict(score=max(score, 0), level=level, flags=flags)

def config_horizon_weeks():
    from . import config
    return list(config.HORIZONS.values())

def global_importance(cs: dict) -> pd.DataFrame:
    return pd.DataFrame(cs["importance"], columns=["feature", "importance_MAE_increase", "std"])

def local_explanation(inp, ctx) -> pd.DataFrame:
    """Replace one input at a time by a neutral reference and report the change in 26-week model units.
    This is model sensitivity (what the model uses), NOT a causal effect."""
    from .scenario_engine import predict_per_outlet, build_launch_frame, _featurize
    cs = ctx["coldstart"]
    f0 = build_launch_frame(inp, cs["support"])
    fz = _featurize(f0, cs["ref"])
    def total(frame):
        frame = frame.copy(); frame["pred"] = np.clip(cs["model"].predict(frame[cs["features"]]), 0, None)
        w = frame.store_type.map(inp.outlets).fillna(0)
        return float((frame["pred"] * w)[frame.week < 26].sum() / max(len(inp.regions), 1))
    base = total(fz)
    alt = {
        "Category (vs. average category)": ("category", None),
        "Price positioning (vs. category median)": ("log_price_index", 0.0),
        "Pack size (vs. median)": ("log_pack_size", float(np.median(fz["log_pack_size"]))),
        "Introductory promotion (removed)": (("promo_flag", "discount_pct"), (0, 0)),
        "Launch timing / seasonality (neutral)": (("woy_sin", "woy_cos", "holiday_flag"), (0, 0, 0)),
    }
    rows = []
    for label, (cols, val) in alt.items():
        g = fz.copy()
        if label.startswith("Category"):
            # average prediction across all categories (price index kept relative, so only the category label changes)
            preds = []
            for c in cs["support"]["categories"]:
                h = g.copy(); h["category"] = c; preds.append(total(h))
            new = float(np.mean(preds))
        else:
            if isinstance(cols, tuple):
                for c, v in zip(cols, val): g[c] = v
            else: g[cols] = val
            new = total(g)
        rows.append(dict(factor=label, units_26w_with=base, units_26w_neutral=new, effect_units=base - new,
                         effect_pct=100 * (base - new) / max(new, 1e-9)))
    out = pd.DataFrame(rows).sort_values("effect_units", key=np.abs, ascending=False)
    out["reading"] = np.where(out.effect_units >= 0, "model raises forecast", "model lowers forecast")
    return out

def partial_dependence(cs: dict, feature: str, grid, base_frame: pd.DataFrame) -> pd.DataFrame:
    X = base_frame.copy()
    out = []
    for v in grid:
        X[feature] = v
        out.append((v, float(np.clip(cs["model"].predict(X[cs["features"]]), 0, None).mean())))
    return pd.DataFrame(out, columns=[feature, "avg_predicted_units"])
