"""Launch simulator: unit economics, demand projection, supply constraints, Monte-Carlo hurdle probability.

Evidence labels used throughout:  [Model] cold-start demand model | [Observed-derived] error calibration
from out-of-fold launches | [Assumption] user-entered or manually configured.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict, replace
import numpy as np
import pandas as pd
from . import config
from .data_ingestion import to_base_size
from .feature_engineering import calendar_features
from .marketing_analytics import uplift_from_spend
from .data_validation import validate_launch_inputs

MARKETING_KEYS = ("ad_budget", "digital_budget", "retail_activation_budget", "influencer_budget")
FIXED_KEYS = MARKETING_KEYS + ("trade_promo_budget", "other_fixed_costs")


@dataclass
class LaunchInputs:
    # product
    product_name: str = "New Product"
    category: str = "Biscuits & Snacks"
    subcategory: str = ""
    brand_positioning: str = "mass market"
    differentiation: str = "medium"
    pack_size: float = 100.0
    pack_unit: str = "g"
    packaging_type: str = "Pouch"
    # pricing (consumer shelf price per pack, tax-inclusive)
    price: float = 35.0
    tax_rate: float = 0.12
    mfg_cost: float = 10.0
    packaging_cost: float = 2.0
    dist_cost: float = 1.5
    trade_margin: float = 0.22          # distributor+retailer margin as share of tax-exclusive shelf price
    wastage_pct: float = 0.02           # returns/wastage as share of manufacturer net revenue
    intro_discount: float = 0.0         # consumer-facing discount fraction
    intro_discount_weeks: int = 4
    discount_funded_share: float = 1.0  # share of the discount funded by the manufacturer
    target_margin: float = 0.20
    competitor_price_low: float | None = None    # per 100 g/ml, optional
    competitor_price_high: float | None = None
    reference_price: float | None = None         # anchor price for what-if analysis
    # market
    country: str = "India"
    regions: list = field(default_factory=lambda: ["North", "South"])
    area_type: str = "Urban"
    segment: str = ""
    launch_date: str = "2025-03-03"
    outlets: dict = field(default_factory=lambda: {"Hypermarket": 20, "Supermarket": 80, "Convenience": 150})
    # marketing
    total_launch_budget: float = 2_400_000.0
    ad_budget: float = 700_000.0
    digital_budget: float = 500_000.0
    retail_activation_budget: float = 300_000.0
    trade_promo_budget: float = 300_000.0
    influencer_budget: float = 100_000.0
    other_fixed_costs: float = 300_000.0
    campaign_weeks: int = 8
    max_uplift: float = 0.20            # ASSUMPTION
    half_sat_per_outlet: float = 2000.0  # ASSUMPTION: spend/outlet giving half of max uplift
    post_campaign_retention: float = 0.5  # ASSUMPTION
    # operations
    initial_inventory: float = 120_000.0
    weekly_capacity: float = 25_000.0
    lead_time_weeks: int = 3
    receivable_days: int = 30
    availability: float = 1.0           # shelf availability relative to historical norm (<=1)
    # hurdle & manual what-if assumptions
    target_profit: float = 0.0
    hurdle_weeks: int = 26
    competitor_price_change_pct: float = 0.0
    cross_elasticity: float = 0.3       # ASSUMPTION
    demand_adjustment_pct: float = 0.0  # ASSUMPTION
    currency: str = config.CURRENCY

    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})
    def validate(self): return validate_launch_inputs(self.to_dict())
    def fixed_costs(self): return float(sum(getattr(self, k) for k in FIXED_KEYS))


# ------------------------------------------------------------------ unit economics
def unit_economics(inp: LaunchInputs, discounted: bool = False, cost_mult: float = 1.0) -> dict:
    """Manufacturer-level unit economics.
    shelf price -> remove tax -> remove trade margin (computed on the pre-discount tax-exclusive price) ->
    remove the manufacturer-funded part of the introductory discount. The discount is never also deducted
    from the trade margin, so nothing is counted twice."""
    d = inp.intro_discount if discounted else 0.0
    ex_tax = inp.price / (1 + inp.tax_rate)
    trade = ex_tax * inp.trade_margin
    disc_cost = d * ex_tax * inp.discount_funded_share
    net = ex_tax - trade - disc_cost
    wastage = net * inp.wastage_pct
    var = (inp.mfg_cost + inp.packaging_cost + inp.dist_cost) * cost_mult + wastage
    contrib = net - var
    return dict(consumer_price=inp.price * (1 - d), price_ex_tax=ex_tax, trade_margin_amt=trade, discount_cost=disc_cost,
                net_revenue=net, wastage=wastage, variable_cost=var, contribution=contrib,
                margin_pct=(contrib / net if net > 0 else float("nan")))

def break_even_volume(fixed: float, contribution_per_unit: float):
    return fixed / contribution_per_unit if contribution_per_unit > 0 else float("inf")


# ------------------------------------------------------------------ context (models + pricing evidence)
def load_context() -> dict | None:
    from .forecasting import load_bundle
    cs = load_bundle("coldstart_model")
    if cs is None:
        return None
    p = config.MODELS_DIR / "pricing.json"
    pr = json.loads(p.read_text()) if p.exists() else {}
    el = dict(pr.get("elasticity", {})); src = {k: "simulated panel" for k in el}
    if os.environ.get("FMCG_PRICE_SOURCE", "real").lower() != "simulated":     # prefer REAL scanner-panel estimates where they exist
        from . import real_data
        try:
            ov, labels = real_data.elasticity_overrides()
            el.update(ov); src.update(labels)
        except Exception:
            pass
    return dict(coldstart=cs, elasticity=el, elasticity_source=src, ratio_range=tuple(pr.get("ratio_range", (0.73, 1.10))),
                demo=pr.get("demo", True))


# ------------------------------------------------------------------ demand projection
def build_launch_frame(inp: LaunchInputs, support: dict, weeks: int = config.COLD_START_WEEKS) -> pd.DataFrame:
    anchor = inp.reference_price or inp.price
    base_size, _ = to_base_size(inp.pack_size, inp.pack_unit)
    ld = pd.Timestamp(inp.launch_date)
    rows = []
    chans = [c for c, n in inp.outlets.items() if n > 0]
    for w in range(weeks):
        dt = ld + pd.Timedelta(weeks=w)
        promo = int(inp.intro_discount > 0 and w < inp.intro_discount_weeks)
        eff = anchor * (1 - inp.intro_discount * promo)
        for r in inp.regions:
            for c in chans:
                rows.append(dict(week=w, date=dt, category=inp.category, region=r, store_type=c, price=eff,
                                 pack_size=base_size, promo_flag=promo, discount_pct=inp.intro_discount * 100 * promo,
                                 age_weeks=float(w)))
    df = pd.DataFrame(rows)
    df["holiday_flag"] = df["date"].dt.isocalendar().week.astype(int).isin(config.HOLIDAY_WEEKS).astype(int)
    df = pd.concat([df, calendar_features(df["date"])], axis=1)
    return df

def _featurize(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    df["log_pack_size"] = np.log(df["pack_size"].clip(lower=1e-6))
    ppu = df["price"] / df["pack_size"] * 100
    df["log_price_index"] = np.log((ppu / df["category"].map(ref)).clip(lower=1e-6))
    return df

def predict_per_outlet(inp: LaunchInputs, ctx: dict, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """[Model] mean units per outlet-week by week x channel (averaged over selected regions)."""
    cs = ctx["coldstart"]
    if inp.category not in cs["ref"]:
        raise ValueError(f"Category '{inp.category}' is not in the training data; supported: {sorted(cs['ref'])}")
    bad = [r for r in inp.regions if r not in cs["support"]["regions"]]
    if bad:
        raise ValueError(f"Regions not in training data: {bad}. Supported: {cs['support']['regions']}")
    f = _featurize(build_launch_frame(inp, cs["support"]) if frame is None else frame, cs["ref"])
    f["pred"] = np.clip(cs["model"].predict(f[cs["features"]]), 0, None)
    return f.groupby(["week", "store_type"])["pred"].mean().unstack("store_type").fillna(0.0)

def elasticity_for(inp: LaunchInputs, ctx: dict) -> float:
    e = ctx.get("elasticity", {})
    return float(e.get(inp.category, e.get("pooled", -1.5)))

def weekly_base_demand(inp: LaunchInputs, ctx: dict, weeks: int = config.COLD_START_WEEKS):
    """Unconstrained weekly demand for the whole launch (median-error path), plus per-channel detail."""
    per = predict_per_outlet(inp, ctx)
    out = np.zeros(weeks)
    detail = {}
    for ch, n in inp.outlets.items():
        if n > 0 and ch in per.columns:
            detail[ch] = per[ch].to_numpy()[:weeks] * n
            out += detail[ch]
    anchor = inp.reference_price or inp.price
    price_adj = (inp.price / anchor) ** elasticity_for(inp, ctx)
    comp_adj = (1 + inp.competitor_price_change_pct / 100) ** inp.cross_elasticity
    spend = sum(getattr(inp, k) for k in MARKETING_KEYS)
    u = uplift_from_spend(spend / max(sum(inp.outlets.values()), 1), inp.max_uplift, inp.half_sat_per_outlet)
    mkt = np.array([1 + u * (1.0 if w < inp.campaign_weeks else inp.post_campaign_retention) for w in range(weeks)])
    mult = price_adj * comp_adj * mkt * inp.availability * (1 + inp.demand_adjustment_pct / 100)
    return out * mult, detail, dict(price_adj=price_adj, comp_adj=comp_adj, marketing_uplift=u, elasticity=elasticity_for(inp, ctx))


# ------------------------------------------------------------------ simulation
def _simulate(D: np.ndarray, inp: LaunchInputs, cost_mult: np.ndarray):
    """D: (n, W) unconstrained demand. Returns dict of arrays after supply constraints and economics."""
    n, W = D.shape
    inv = np.full(n, float(inp.initial_inventory))
    sold = np.zeros((n, W)); produced = np.zeros(W)
    for w in range(W):
        arr = inp.weekly_capacity if w >= inp.lead_time_weeks else 0.0
        produced[w] = arr
        avail = inv + arr
        s = np.minimum(D[:, w], avail)
        sold[:, w] = s
        inv = avail - s
    promo_w = np.array([inp.intro_discount > 0 and w < inp.intro_discount_weeks for w in range(W)])
    ue_reg, ue_pro = unit_economics(inp, False), unit_economics(inp, True)
    net_w = np.where(promo_w, ue_pro["net_revenue"], ue_reg["net_revenue"])
    waste_w = net_w * inp.wastage_pct
    vc = (inp.mfg_cost + inp.packaging_cost + inp.dist_cost)
    contrib_unit = net_w[None, :] - vc * cost_mult[:, None] - waste_w[None, :]
    contrib = sold * contrib_unit
    cons_price = np.where(promo_w, ue_pro["consumer_price"], ue_reg["consumer_price"])
    return dict(sold=sold, lost=D - sold, contrib=contrib, net_rev=sold * net_w[None, :], sell_through=sold * cons_price[None, :],
                cum_after_fixed=np.cumsum(contrib, axis=1) - inp.fixed_costs(), end_inv=inv, produced=produced, net_w=net_w)

def _log_ratio_draws(cs: dict, n: int, rng) -> np.ndarray:
    """Smoothed bootstrap of out-of-fold product-level log(actual/predicted) 26-week totals."""
    lr = np.asarray(cs["log_ratio"])
    bw = 1.06 * lr.std(ddof=1) * len(lr) ** (-0.2)
    return rng.choice(lr, n) + rng.normal(0, bw, n)

def run_assessment(inp: LaunchInputs, ctx: dict, n_sims: int = 2000, seed: int = config.SEED, mc: bool = True) -> dict:
    v = inp.validate()
    if v["errors"]:
        raise ValueError("; ".join(v["errors"]))
    cs = ctx["coldstart"]
    W = config.COLD_START_WEEKS
    base_D, detail, adj = weekly_base_demand(inp, ctx, W)
    rng = np.random.default_rng(seed)
    if mc:
        m = np.exp(_log_ratio_draws(cs, n_sims, rng))
        sd = cs["weekly_noise_sd"]
        eps = rng.normal(-sd ** 2 / 2, sd, (n_sims, W))
        cost_sd = 0.03   # ASSUMPTION: unit-cost realisation uncertainty
        cm = np.clip(rng.normal(1.0, cost_sd, n_sims), 0.8, 1.3)
        D = base_D[None, :] * m[:, None] * np.exp(eps)
    else:
        n_sims = 1
        m = np.array([np.exp(np.median(cs["log_ratio"]))]); cm = np.ones(1)
        D = base_D[None, :] * m[:, None]
    S = _simulate(D, inp, cm)
    hw = min(inp.hurdle_weeks, W)
    horizons = []
    for name, wk in config.HORIZONS.items():
        u = S["sold"][:, :wk].sum(1); rv = S["net_rev"][:, :wk].sum(1); c = S["contrib"][:, :wk].sum(1)
        horizons.append(dict(horizon=name, weeks=wk, units_mean=u.mean(), units_p10=np.percentile(u, 10), units_p50=np.percentile(u, 50),
                             units_p90=np.percentile(u, 90), net_revenue_mean=rv.mean(), contribution_mean=c.mean(),
                             contribution_p10=np.percentile(c, 10), contribution_p90=np.percentile(c, 90),
                             after_launch_costs_mean=c.mean() - inp.fixed_costs()))
    cum = S["cum_after_fixed"]
    p_hurdle = float((cum[:, hw - 1] >= inp.target_profit).mean())
    tot_c, tot_r = S["contrib"][:, :hw].sum(1), S["net_rev"][:, :hw].sum(1)
    ue_reg = unit_economics(inp, False)
    # base path (median demand multiplier) for break-even time, cash and inventory
    base = _simulate(base_D[None, :] * np.exp(np.median(cs["log_ratio"])), inp, np.ones(1))
    be_week = next((w + 1 for w in range(W) if base["cum_after_fixed"][0, w] >= 0), None)
    hours = dict(
        break_even_units_regular=break_even_volume(inp.fixed_costs(), ue_reg["contribution"]),
        break_even_revenue_regular=break_even_volume(inp.fixed_costs(), ue_reg["contribution"]) * ue_reg["net_revenue"],
        break_even_units_blended=break_even_volume(inp.fixed_costs(), base["contrib"].sum() / max(base["sold"].sum(), 1e-9)),
        break_even_week_base=be_week,
        p_break_even_12m=float((cum[:, W - 1] >= 0).mean()),
        required_hurdle_units=break_even_volume(inp.fixed_costs() + inp.target_profit, ue_reg["contribution"]))
    # inventory
    lt = max(int(inp.lead_time_weeks), 1)
    cover = D[:, :lt].sum(1)
    inv = dict(planned_initial=inp.initial_inventory, recommended_initial_p90=float(np.percentile(cover, 90)),
               demand_until_replenishment_p50=float(np.percentile(cover, 50)),
               p_stockout=float((S["lost"][:, :hw].sum(1) > 0.01 * D[:, :hw].sum(1)).mean()),
               expected_lost_units=float(S["lost"][:, :hw].sum(1).mean()),
               p_overstock=float((S["end_inv"] > 4 * np.maximum(S["sold"][:, -4:].mean(1), 1e-9)).mean()),
               expected_end_inventory=float(S["end_inv"].mean()))
    # cash (base path)
    var_unit_prod = inp.mfg_cost + inp.packaging_cost
    delay = int(np.ceil(inp.receivable_days / 7))
    cash = np.zeros(W + delay + 1)
    cash[0] -= inp.fixed_costs() + inp.initial_inventory * var_unit_prod
    for w in range(W):
        cash[w] -= base["produced"][w] * var_unit_prod
        cash[w] -= base["sold"][0, w] * (inp.dist_cost + base["net_w"][w] * inp.wastage_pct)
        cash[w + delay] += base["net_rev"][0, w]
    cumcash = np.cumsum(cash)
    wc = dict(peak_funding_need=float(-cumcash.min()) if cumcash.min() < 0 else 0.0, week_of_peak=int(cumcash.argmin()),
              cash_at_12m=float(cumcash[W - 1]))
    spent = inp.fixed_costs() - inp.other_fixed_costs
    budget = dict(planned_marketing_spend=float(spent), total_budget=inp.total_launch_budget,
                  utilisation_pct=float(100 * spent / inp.total_launch_budget) if inp.total_launch_budget else None)
    # weekly table
    qs = lambda a: np.percentile(a, [10, 50, 90], axis=0)
    ps, pd_ = qs(S["sold"]), qs(D)
    weekly = pd.DataFrame(dict(week=np.arange(1, W + 1), date=[pd.Timestamp(inp.launch_date) + pd.Timedelta(weeks=w) for w in range(W)],
                               demand_p10=pd_[0], demand_p50=pd_[1], demand_p90=pd_[2], sold_p10=ps[0], sold_p50=ps[1], sold_p90=ps[2],
                               sold_mean=S["sold"].mean(0), cum_after_fixed_mean=cum.mean(0),
                               cum_after_fixed_p10=np.percentile(cum, 10, 0), cum_after_fixed_p90=np.percentile(cum, 90, 0)))
    for ch, a in detail.items():
        weekly[f"base_{ch}"] = a
    from .explainability import assess_reliability
    rel = assess_reliability(inp, ctx)
    res = dict(inputs=inp.to_dict(), unit_economics=dict(regular=ue_reg, intro=unit_economics(inp, True)),
               adjustments=adj, horizons=pd.DataFrame(horizons), p_hurdle=p_hurdle, hurdle_weeks=hw,
               margin_pct_hurdle_window=float(np.mean(tot_c / np.maximum(tot_r, 1e-9))),
               expected_after_launch_costs=float(tot_c.mean() - inp.fixed_costs()), breakeven=hours, inventory=inv,
               working_capital=wc, budget=budget, weekly=weekly, reliability=rel, n_sims=n_sims,
               scenarios=scenario_table(inp, ctx, base_D) if mc else None,
               error_model=dict(n_products=int(cs["n_ratio_products"]), log_ratio_sd=float(np.std(cs["log_ratio"])),
                                log_ratio_median=float(np.median(cs["log_ratio"])), weekly_noise_sd=float(cs["weekly_noise_sd"])))
    res["decision"] = None
    return res


# ------------------------------------------------------------------ scenarios (conservative / base / optimistic)
SCENARIO_ASSUMPTIONS = {
    "Conservative": dict(quantile=10, cost_mult=1.05, availability_mult=0.90),
    "Base": dict(quantile=50, cost_mult=1.00, availability_mult=1.00),
    "Optimistic": dict(quantile=90, cost_mult=0.97, availability_mult=1.00),
}

def scenario_table(inp: LaunchInputs, ctx: dict, base_D: np.ndarray) -> pd.DataFrame:
    """Demand level = empirical quantile of out-of-fold product-level log-ratios (observed-derived);
    cost and availability adjustments are MANUAL assumptions shown in the table."""
    cs = ctx["coldstart"]; lr = np.asarray(cs["log_ratio"]); W = len(base_D)
    rows = []
    for name, a in SCENARIO_ASSUMPTIONS.items():
        m = np.exp(np.percentile(lr, a["quantile"]))
        D = (base_D * m * a["availability_mult"])[None, :]
        S = _simulate(D, inp, np.array([a["cost_mult"]]))
        hw = min(inp.hurdle_weeks, W)
        for_h = {}
        for hname, wk in (("3m", 13), ("6m", 26), ("12m", 52)):
            for_h[f"units_{hname}"] = float(S["sold"][0, :wk].sum())
        rev26, c26 = S["net_rev"][0, :26].sum(), S["contrib"][0, :26].sum()
        rows.append(dict(scenario=name, demand_quantile=f"P{a['quantile']}", demand_multiplier=m, cost_multiplier=a["cost_mult"],
                         availability_multiplier=a["availability_mult"], **for_h, net_revenue_6m=float(rev26),
                         variable_cost_6m=float(rev26 - c26), contribution_6m=float(c26),
                         after_launch_costs_6m=float(c26 - inp.fixed_costs()),
                         contribution_margin_6m=float(c26 / rev26) if rev26 > 0 else np.nan,
                         meets_hurdle=bool(S["cum_after_fixed"][0, hw - 1] >= inp.target_profit),
                         lost_units_6m=float(S["lost"][0, :26].sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what-if & sensitivity
def apply_whatif(inp: LaunchInputs, **kw) -> LaunchInputs:
    """kw: price_pct, mfg_cost_pct, marketing_pct, outlets_mult, delay_weeks, competitor_price_pct, demand_pct, discount."""
    d = inp.to_dict()
    d["reference_price"] = inp.reference_price or inp.price
    if "price_pct" in kw: d["price"] = inp.price * (1 + kw["price_pct"] / 100)
    if "mfg_cost_pct" in kw: d["mfg_cost"] = inp.mfg_cost * (1 + kw["mfg_cost_pct"] / 100)
    if "marketing_pct" in kw:
        for k in MARKETING_KEYS + ("trade_promo_budget",): d[k] = getattr(inp, k) * (1 + kw["marketing_pct"] / 100)
        d["total_launch_budget"] = max(inp.total_launch_budget, sum(d[k] for k in MARKETING_KEYS + ("trade_promo_budget",)))
    if "outlets_mult" in kw: d["outlets"] = {k: int(round(v * kw["outlets_mult"])) for k, v in inp.outlets.items()}
    if "delay_weeks" in kw: d["launch_date"] = str((pd.Timestamp(inp.launch_date) + pd.Timedelta(weeks=kw["delay_weeks"])).date())
    if "competitor_price_pct" in kw: d["competitor_price_change_pct"] = inp.competitor_price_change_pct + kw["competitor_price_pct"]
    if "demand_pct" in kw: d["demand_adjustment_pct"] = inp.demand_adjustment_pct + kw["demand_pct"]
    if "discount" in kw: d["intro_discount"] = kw["discount"]
    return LaunchInputs.from_dict(d)

STANDARD_WHATIFS = {
    "Price -5%": dict(price_pct=-5), "Price +10%": dict(price_pct=10), "Marketing budget +20%": dict(marketing_pct=20),
    "Distribution coverage x2": dict(outlets_mult=2), "Launch delayed 4 weeks": dict(delay_weeks=4),
    "Competitor prices -10%": dict(competitor_price_pct=-10), "Demand 15% below forecast": dict(demand_pct=-15),
    "Manufacturing cost +10%": dict(mfg_cost_pct=10),
}

def what_if_table(inp: LaunchInputs, ctx: dict, base: dict | None = None, n_sims: int = 600) -> pd.DataFrame:
    base = base or run_assessment(inp, ctx, n_sims=n_sims, mc=True)
    rows = [dict(scenario="Current plan", p_hurdle=base["p_hurdle"], expected_after_launch_costs=base["expected_after_launch_costs"],
                 units_6m=float(base["horizons"].set_index("horizon").loc["6 months", "units_mean"]), note="")]
    for name, kw in STANDARD_WHATIFS.items():
        try:
            r = run_assessment(apply_whatif(inp, **kw), ctx, n_sims=n_sims)
            rows.append(dict(scenario=name, p_hurdle=r["p_hurdle"], expected_after_launch_costs=r["expected_after_launch_costs"],
                             units_6m=float(r["horizons"].set_index("horizon").loc["6 months", "units_mean"]),
                             note="price effect = within-product elasticity (observational)" if "Price" in name else ""))
        except ValueError as e:
            rows.append(dict(scenario=name, p_hurdle=np.nan, expected_after_launch_costs=np.nan, units_6m=np.nan, note=str(e)))
    t = pd.DataFrame(rows)
    t["delta_p_hurdle_pts"] = 100 * (t.p_hurdle - t.p_hurdle.iloc[0])
    t["delta_after_launch_costs"] = t.expected_after_launch_costs - t.expected_after_launch_costs.iloc[0]
    return t

def sensitivity_tornado(inp: LaunchInputs, ctx: dict, pct: float = 10.0) -> pd.DataFrame:
    """One-at-a-time +/- pct change; metric = expected contribution after launch costs (median-demand path)."""
    def metric(i):
        r = run_assessment(i, ctx, mc=False)
        return r["expected_after_launch_costs"]
    base = metric(inp)
    specs = {"Shelf price": "price_pct", "Manufacturing cost": "mfg_cost_pct", "Marketing budget": "marketing_pct",
             "Distribution (outlets)": "outlets_mult", "Demand level": "demand_pct"}
    rows = []
    for label, key in specs.items():
        vals = []
        for s in (-1, 1):
            kw = {key: (1 + s * pct / 100) if key == "outlets_mult" else s * pct}
            try: vals.append(metric(apply_whatif(inp, **kw)) - base)
            except ValueError: vals.append(np.nan)
        rows.append(dict(driver=label, low=vals[0], high=vals[1], swing=np.nanmax(np.abs(vals))))
    return pd.DataFrame(rows).sort_values("swing", ascending=False)

def price_curve(inp: LaunchInputs, ctx: dict, grid=None) -> pd.DataFrame:
    from .pricing_analytics import price_response_curve
    base_D, _, _ = weekly_base_demand(inp, ctx)
    wk = base_D[:inp.hurdle_weeks].sum()
    f = lambda p: unit_economics(replace(inp, price=p))["contribution"]
    return price_response_curve(inp.price, wk, elasticity_for(inp, ctx), f, ctx["ratio_range"], grid)
