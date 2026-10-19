"""Exact TreeSHAP for the cold-start LightGBM (via LightGBM's native `pred_contrib`; no extra dependency).

Contributions are in LOG-demand space (the Poisson link): base + sum(contributions) = log(predicted units per outlet-week),
so exp(contribution) is a multiplicative effect on demand. One-hot columns are summed back to their source feature."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import scenario_engine as se
from .feature_engineering import ppu100  # noqa: F401  (kept for parity with featurisation)


def _parts(cs):
    pipe = cs["model"]
    if not hasattr(pipe, "named_steps") or not hasattr(pipe.named_steps.get("m"), "booster_"):
        return None
    return pipe.named_steps["p"], pipe.named_steps["m"]


def supported(ctx: dict) -> bool:
    return _parts(ctx["coldstart"]) is not None


def explain_rows(frame: pd.DataFrame, ctx: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (contributions by ORIGINAL feature per row, base value per row, raw log-prediction per row)."""
    cs = ctx["coldstart"]; parts = _parts(cs)
    if parts is None:
        raise ValueError("SHAP needs the LightGBM cold-start model; the saved model is a different type.")
    pre, est = parts
    X = pre.transform(se._featurize(frame, cs["ref"])[cs["features"]])
    names = list(pre.get_feature_names_out())
    C = est.booster_.predict(X, pred_contrib=True)
    base, contrib = C[:, -1], C[:, :-1]
    src = [n.split("__", 1)[1] for n in names]
    src = [next((f for f in cs["features"] if s == f or s.startswith(f + "_")), s) for s in src]
    df = pd.DataFrame(contrib, columns=names).T.groupby(src).sum().T
    return df, base, base + contrib.sum(1)


def explain_launch(inp: se.LaunchInputs, ctx: dict) -> dict:
    cs = ctx["coldstart"]
    frame = se.build_launch_frame(inp, cs["support"])
    df, base, raw = explain_rows(frame, ctx)
    mean = df.mean()
    out = pd.DataFrame(dict(feature=mean.index, mean_shap=mean.values, mean_abs_shap=df.abs().mean().values,
                            multiplier=np.exp(mean.values))).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return dict(table=out, base=float(base.mean()), prediction_log=float(raw.mean()), by_week=df.groupby(frame["week"].values).mean(),
                additivity_error=float(np.abs(base + df.sum(1).to_numpy() - raw).max()))
