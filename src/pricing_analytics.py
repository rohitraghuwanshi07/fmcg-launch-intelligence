"""Model C: price response. OBSERVATIONAL within-series estimates, not causal elasticities."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

CAVEAT = ("Observational estimate: identified from within-product, within-store price variation after "
          "controlling for promotions and week fixed effects. Prices were not randomised, so this is an "
          "association / scenario input, not a proven causal elasticity.")

def _two_way_demean(df: pd.DataFrame, cols, g1, g2, iters=15):
    out = df[cols].astype(float).copy()
    for _ in range(iters):
        out = out - out.groupby(df[g1].values).transform("mean")
        out = out - out.groupby(df[g2].values).transform("mean")
    return out

def estimate_elasticity(panel: pd.DataFrame, min_age: float = 20, min_rows: int = 800) -> pd.DataFrame:
    d = panel[panel.units > 0].copy()
    if "weeks_since_intro" in d:
        d = d[d.weeks_since_intro.isna() | (d.weeks_since_intro >= min_age)]  # exclude launch ramp-up
    d["series"] = d["product_id"].astype(str) + "|" + d["store_id"].astype(str)
    d["ly"], d["lp"] = np.log(d["units"]), np.log(d["price"])
    d["date_key"] = d["date"].astype(str)
    rows = []
    scopes = [("All categories (pooled)", d)] + [(c, g) for c, g in d.groupby("category")]
    for name, g in scopes:
        if len(g) < min_rows or g.series.nunique() < 5:
            rows.append(dict(scope=name, elasticity=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, n_obs=len(g),
                             n_products=g.product_id.nunique(), residual_price_sd=np.nan, note="Insufficient data"))
            continue
        dm = _two_way_demean(g, ["ly", "lp", "promo_flag"], "series", "date_key")
        X = dm[["lp", "promo_flag"]]
        fit = sm.OLS(dm["ly"], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g["product_id"])[0]})
        b, se = float(fit.params["lp"]), float(fit.bse["lp"])
        rows.append(dict(scope=name, elasticity=b, se=se, ci_low=b - 1.96 * se, ci_high=b + 1.96 * se, n_obs=len(g),
                         n_products=g.product_id.nunique(), residual_price_sd=float(dm["lp"].std()),
                         promo_coef=float(fit.params["promo_flag"]), note=CAVEAT if name.startswith("All") else ""))
    return pd.DataFrame(rows)

def supported_price_ratio_range(panel: pd.DataFrame) -> tuple[float, float]:
    """2nd-98th percentile of price relative to the series median: range with observed variation."""
    r = panel["price"] / panel.groupby(["product_id", "store_id"])["price"].transform("median")
    return float(r.quantile(0.02)), float(r.quantile(0.98))

def price_band_table(panel: pd.DataFrame, n_bands: int = 5) -> pd.DataFrame:
    d = panel.copy()
    from .feature_engineering import add_price_index, fit_category_reference
    d = add_price_index(d, fit_category_reference(d))
    d["band"] = pd.qcut(d["price_index"], n_bands, duplicates="drop")
    t = d.groupby("band", observed=True).agg(avg_units=("units", "mean"), rows=("units", "size"),
                                              avg_price_index=("price_index", "mean")).reset_index()
    t["band"] = t["band"].astype(str)
    return t

def price_response_curve(base_price, base_units_weekly_total, elasticity, contribution_fn, ratio_range, grid=None):
    """Constant-elasticity scenario curve anchored at base_price. contribution_fn(price)->contribution/unit."""
    grid = np.linspace(0.7, 1.3, 25) if grid is None else grid
    rows = []
    for r in grid:
        p = base_price * r
        u = base_units_weekly_total * r ** elasticity
        c = contribution_fn(p)
        rows.append(dict(price_ratio=r, price=p, units=u, revenue=u * p, contribution=u * c,
                         supported=bool(ratio_range[0] <= r <= ratio_range[1])))
    return pd.DataFrame(rows)
