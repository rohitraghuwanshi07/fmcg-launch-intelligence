"""Model E: marketing analytics. Observational promo effect, campaign-upload analysis, scenario response curve."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .pricing_analytics import _two_way_demean

CAMPAIGN_COLS = ["date", "channel", "spend", "impressions", "clicks", "conversions"]
CAMPAIGN_TEMPLATE = pd.DataFrame([["2024-06-03", "Digital - social", 50000, 1200000, 24000, 900, 180000]],
                                 columns=CAMPAIGN_COLS + ["attributed_revenue"])

def promo_effect(panel: pd.DataFrame) -> dict:
    d = panel[panel.units > 0].copy()
    d["series"] = d.product_id.astype(str) + "|" + d.store_id.astype(str)
    d["ly"], d["dk"] = np.log(d.units), d.date.astype(str)
    dm = _two_way_demean(d, ["ly", "promo_flag"], "series", "dk")
    fit = sm.OLS(dm.ly, dm[["promo_flag"]]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.product_id)[0]})
    b, se = float(fit.params.iloc[0]), float(fit.bse.iloc[0])
    return dict(log_uplift=b, uplift_pct=100 * (np.exp(b) - 1), ci_low_pct=100 * (np.exp(b - 1.96 * se) - 1),
                ci_high_pct=100 * (np.exp(b + 1.96 * se) - 1), n_obs=len(d),
                caveat="Observational association (promotions were not randomised; includes the price effect of the discount).")

def validate_campaign_upload(df: pd.DataFrame) -> dict:
    miss = [c for c in CAMPAIGN_COLS if c not in df.columns]
    if miss:
        return dict(errors=[f"Missing columns: {miss}"], warnings=[])
    e = []
    for c in ("spend", "impressions", "clicks", "conversions"):
        if pd.to_numeric(df[c], errors="coerce").isna().any() or (pd.to_numeric(df[c], errors="coerce") < 0).any():
            e.append(f"Column '{c}' has non-numeric or negative values.")
    if pd.to_datetime(df["date"], errors="coerce").isna().any():
        e.append("Some dates are invalid.")
    return dict(errors=e, warnings=[])

def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("channel").agg(spend=("spend", "sum"), impressions=("impressions", "sum"),
                                  clicks=("clicks", "sum"), conversions=("conversions", "sum"),
                                  **({"revenue": ("attributed_revenue", "sum")} if "attributed_revenue" in df else {}))
    g["CTR_pct"] = 100 * g.clicks / g.impressions.replace(0, np.nan)
    g["CPC"] = g.spend / g.clicks.replace(0, np.nan)
    g["CPA"] = g.spend / g.conversions.replace(0, np.nan)
    if "revenue" in g:
        g["ROAS"] = g.revenue / g.spend.replace(0, np.nan)
    return g.reset_index()

def uplift_from_spend(spend_per_outlet: float, max_uplift: float, half_sat: float) -> float:
    """ASSUMPTION-BASED saturating response: uplift = max * (1 - 2^(-spend/half_sat)). Not estimated from data."""
    if half_sat <= 0: return 0.0
    return float(max_uplift * (1 - 2 ** (-max(spend_per_outlet, 0) / half_sat)))

def suggest_channel_mix(positioning: str, segment_hint: str, urban: str) -> pd.DataFrame:
    """Rule-based starting split of the marketing budget (editable). Not a trained model."""
    base = {"Digital / social": 0.25, "Retail activation (in-store)": 0.25, "Trade promotion": 0.25,
            "Mass media (TV/print)": 0.15, "Influencer / creator": 0.10}
    if positioning in ("premium", "super-premium"):
        base["Digital / social"] += 0.10; base["Influencer / creator"] += 0.05; base["Trade promotion"] -= 0.10; base["Mass media (TV/print)"] -= 0.05
    if positioning == "economy":
        base["Trade promotion"] += 0.10; base["Retail activation (in-store)"] += 0.05; base["Influencer / creator"] -= 0.05; base["Digital / social"] -= 0.10
    if str(urban).lower() == "rural":
        base["Mass media (TV/print)"] += 0.10; base["Digital / social"] -= 0.10
    if "deal" in str(segment_hint).lower():
        base["Trade promotion"] += 0.05; base["Mass media (TV/print)"] -= 0.05
    s = sum(base.values())
    return pd.DataFrame([(k, round(v / s, 3)) for k, v in base.items()], columns=["channel", "share"])
