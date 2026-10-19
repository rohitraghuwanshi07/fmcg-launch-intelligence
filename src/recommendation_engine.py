"""Rule-based recommendation engine. Every recommendation cites evidence from THIS launch's results and
states whether it is model-derived, rule-based or manual. Nothing here is a causal claim."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from . import config
from . import scenario_engine as se

P_LAUNCH, P_PILOT = 0.70, 0.40   # documented decision thresholds

@dataclass
class Recommendation:
    area: str
    action: str
    reason: str
    evidence: str
    benefit: str
    tradeoff: str
    confidence: str
    source: str   # model-derived | rule-based | manual-assumption
    def to_dict(self): return asdict(self)

def decide(res: dict) -> dict:
    inp = res["inputs"]; p = res["p_hurdle"]; ue = res["unit_economics"]["regular"]
    lvl = res["reliability"]["level"]; sd = res["error_model"]["log_ratio_sd"]
    rule = (f"Launch if P(hurdle) >= {P_LAUNCH:.0%}, unit margin >= target and model reliability is not Low; "
            f"Pilot if P(hurdle) >= {P_PILOT:.0%} or reliability is Low / forecast error is large; "
            "Modify or Reposition if P(hurdle) is below that but unit contribution is positive; Reconsider if unit contribution <= 0.")
    if ue["contribution"] <= 0:
        d, why = "Reconsider the launch", "Contribution per unit is not positive, so volume cannot fix the economics."
    elif p >= P_LAUNCH and ue["margin_pct"] >= inp["target_margin"] and lvl != "Low" and sd <= 0.6:
        d, why = "Launch", f"P(hurdle) = {p:.0%} with unit margin {ue['margin_pct']:.0%} and acceptable model reliability."
    elif p >= P_LAUNCH:
        d, why = "Run a limited regional pilot", (f"P(hurdle) = {p:.0%} looks favourable, but " + (
            "model reliability is Low (see risks)" if lvl == "Low" else "unit margin is below target" if ue["margin_pct"] < inp["target_margin"] else "forecast error is large") +
            ", so validate demand on a small scale first.")
    elif p >= P_PILOT:
        d, why = ("Modify the plan, then pilot" if ue["margin_pct"] < inp["target_margin"] else "Run a limited regional pilot"), (
            f"P(hurdle) = {p:.0%} is borderline; improve economics or reduce exposure before scaling.")
    else:
        d, why = "Reposition or modify before launching", f"P(hurdle) = {p:.0%} is low; changes to price, cost, scale or channel are needed."
    return dict(decision=d, rationale=why, rule=rule)

def _fmt(x): return f"{config.CURRENCY} {x:,.0f}"

def required_price_for_margin(inp: se.LaunchInputs) -> float:
    cost = inp.mfg_cost + inp.packaging_cost + inp.dist_cost
    net = cost / (1 - inp.wastage_pct - inp.target_margin)
    return net / (1 - inp.trade_margin) * (1 + inp.tax_rate)

def generate(res: dict, ctx: dict, market: dict | None = None, segments: pd.DataFrame | None = None) -> list[Recommendation]:
    inp = se.LaunchInputs.from_dict(res["inputs"])
    ue, ue_p = res["unit_economics"]["regular"], res["unit_economics"]["intro"]
    H = res["horizons"].set_index("horizon"); inv = res["inventory"]; be = res["breakeven"]
    lvl = res["reliability"]["level"]
    conf_model = "Low" if lvl == "Low" else "Medium"
    recs: list[Recommendation] = []

    # 1. unit economics / pricing floor
    if ue["margin_pct"] < inp.target_margin:
        rp = required_price_for_margin(inp)
        recs.append(Recommendation("Pricing / cost", f"Raise shelf price to about {config.CURRENCY} {rp:.0f} or cut unit cost / trade margin",
            f"Contribution margin {ue['margin_pct']:.0%} is below the {inp.target_margin:.0%} target.",
            f"Net revenue/unit {ue['net_revenue']:.2f}; variable cost/unit {ue['variable_cost']:.2f}; contribution {ue['contribution']:.2f}.",
            "Meets the target margin on every unit sold (arithmetic, before demand response).",
            "Higher price lowers volume (see price-response analysis) and may push positioning above comparables.", "High", "rule-based"))
    # 2. price optimisation within the supported range (elasticity-based scenario)
    try:
        pc = se.price_curve(inp, ctx)
        ok = pc[pc.supported]
        if len(ok):
            best = ok.loc[ok.contribution.idxmax()]
            cur = pc.iloc[(pc.price_ratio - 1).abs().argmin()]
            gain = best.contribution - cur.contribution
            if abs(best.price_ratio - 1) >= 0.05 and gain > 0.03 * abs(cur.contribution):
                recs.append(Recommendation("Pricing", f"Test a shelf price near {best.price:.0f} ({(best.price_ratio - 1):+.0%} vs plan)",
                    "The category's within-product price response implies a higher gross contribution at that price.",
                    f"Elasticity {se.elasticity_for(inp, ctx):.2f} (observational); gross contribution over {inp.hurdle_weeks}w moves by {_fmt(gain)} before launch costs; within the observed price range.",
                    f"Up to {_fmt(gain)} more contribution (scenario estimate, not a guarantee).",
                    "Elasticity is associational, not causal; new products may respond differently. Validate with a price test.", conf_model, "model-derived"))
    except Exception:
        pass
    # 3. competitor positioning
    if market and market.get("available"):
        pct = market["percentile"]
        if pct >= 85 and inp.differentiation != "high":
            recs.append(Recommendation("Positioning", "Strengthen differentiation or lower the entry price / launch with a smaller pack",
                f"Price per 100 units sits at the {pct:.0f}th percentile of {market['n_comparables']} comparables without strong differentiation.",
                f"Median comparable {market['median_ppu100']:.1f} per 100; yours is x{market['index_vs_median']:.2f} the median.",
                "Reduces price-related adoption risk.", "Lower price compresses margin.", "Medium", "rule-based"))
        elif pct <= 15 and ue["margin_pct"] < inp.target_margin + 0.05:
            recs.append(Recommendation("Positioning", "Value-priced entry is thin on margin: consider a modest price increase or premium sub-variant",
                f"Price sits at the {pct:.0f}th percentile of comparables while margin is {ue['margin_pct']:.0%}.",
                f"Comparable range per 100: {market['q25']:.1f}-{market['q75']:.1f} (IQR).", "Recovers margin headroom.", "May lose price-led volume.", "Medium", "rule-based"))
        if market["n_comparables"] >= 8 and inp.differentiation == "low":
            recs.append(Recommendation("Competitive differentiation", "Develop a clear differentiator (format, benefit, claims) before launch",
                f"{market['n_comparables']} comparable products exist and differentiation is rated low.",
                "Count of comparables is a crowding proxy only (not market share).", "Lower risk of price-only competition.", "Development time and cost.", "Medium", "rule-based"))
    # 4. break-even vs plausible demand
    need = be["required_hurdle_units"]
    hw_row = H.loc[[k for k, v in config.HORIZONS.items() if v >= inp.hurdle_weeks][0]] if inp.hurdle_weeks <= 52 else H.iloc[-1]
    if np.isfinite(need):
        if need > hw_row.units_p90:
            recs.append(Recommendation("Scale / cost structure", "Reduce fixed launch spend or restructure costs; the hurdle is above plausible demand",
                f"Units needed ({need:,.0f}) exceed the P90 demand ({hw_row.units_p90:,.0f}) over {inp.hurdle_weeks} weeks.",
                f"P10/P50/P90 units: {hw_row.units_p10:,.0f} / {hw_row.units_p50:,.0f} / {hw_row.units_p90:,.0f}.",
                "Brings the hurdle within the modelled demand range.", "A smaller launch may reduce reach.", "Medium", "model-derived"))
        elif need > hw_row.units_p50:
            recs.append(Recommendation("Risk mitigation", "Stage the investment: commit part of the budget and release the rest after early sell-through checks",
                f"Units needed ({need:,.0f}) are above the median outcome ({hw_row.units_p50:,.0f}).",
                f"P(hurdle) = {res['p_hurdle']:.0%}.", "Caps downside if demand lands near the median.", "Slower build-up; competitors may react.", "Medium", "rule-based"))
    # 5. launch timing
    try:
        outs = {w: se.run_assessment(se.apply_whatif(inp, delay_weeks=w), ctx, mc=False)["expected_after_launch_costs"] for w in (0, 4, 8, 13, 26)}
        bw = max(outs, key=outs.get)
        if bw != 0 and outs[bw] - outs[0] > 0.05 * abs(outs[0]) + 1:
            recs.append(Recommendation("Launch timing", f"Consider delaying launch by about {bw} weeks",
                "The demand model's seasonal pattern favours a later start.",
                f"Median-path contribution after launch costs: {_fmt(outs[0])} now vs {_fmt(outs[bw])} with a {bw}-week delay.",
                f"About {_fmt(outs[bw] - outs[0])} (scenario estimate).", "Delay carries competitive and cash-flow costs not modelled here.", conf_model, "model-derived"))
    except Exception:
        pass
    # 6. inventory
    if inv["p_stockout"] > 0.25:
        recs.append(Recommendation("Inventory planning", f"Raise initial inventory toward {inv['recommended_initial_p90']:,.0f} units or bring forward capacity / shorten lead time",
            f"{inv['p_stockout']:.0%} of simulated outcomes lose more than 1% of demand to stock-outs.",
            f"Planned {inv['planned_initial']:,.0f} units; P90 demand before replenishment {inv['recommended_initial_p90']:,.0f}; expected lost units {inv['expected_lost_units']:,.0f}.",
            "Protects sales during ramp-up.", "More working capital tied up and greater overstock risk.", "Medium", "model-derived"))
    elif inv["p_overstock"] > 0.4:
        recs.append(Recommendation("Inventory planning", "Reduce the opening stock build or phase production",
            f"{inv['p_overstock']:.0%} of simulated outcomes end the year with more than 4 weeks of stock.",
            f"Expected end inventory {inv['expected_end_inventory']:,.0f} units.", "Lower write-off and cash exposure.", "Higher stock-out risk if demand surprises upward.", "Medium", "model-derived"))
    # 7. marketing intensity
    rev26 = H.loc["6 months", "net_revenue_mean"]
    mk = res["budget"]["planned_marketing_spend"]
    if rev26 > 0 and mk / rev26 > 0.35:
        recs.append(Recommendation("Advertising", "Trim or phase advertising; spend is heavy relative to expected 6-month net revenue",
            f"Launch spend is {mk / rev26:.0%} of expected 6-month manufacturer net revenue.",
            f"Spend {_fmt(mk)} vs revenue {_fmt(rev26)}. Marketing response is an ASSUMPTION (max uplift {inp.max_uplift:.0%}, half-saturation {inp.half_sat_per_outlet:,.0f}/outlet).",
            "Improves return if the true response saturates early.", "Lower awareness if the true response is stronger.", "Low", "manual-assumption"))
    elif res["adjustments"]["marketing_uplift"] < 0.5 * inp.max_uplift:
        recs.append(Recommendation("Advertising", "Run a controlled regional test before deciding on a larger campaign",
            "Assumed marketing response is still low per outlet.", f"Modelled uplift {res['adjustments']['marketing_uplift']:.0%} of baseline demand.",
            "Measures real incremental sales instead of relying on an assumed curve.", "Delays scale-up.", "Low", "manual-assumption"))
    # 8. introductory promotion
    if inp.intro_discount > 0 and ue_p["contribution"] <= 0.25 * ue["contribution"]:
        recs.append(Recommendation("Trade promotions", "Shorten or share the funding of the introductory discount",
            "Promo-period unit contribution is under a quarter of regular contribution.",
            f"Contribution/unit {ue_p['contribution']:.2f} (promo) vs {ue['contribution']:.2f} (regular).", "Protects margin.", "Smaller trial uplift.", "High", "rule-based"))
    # 9. target consumer
    if segments is not None and len(segments):
        s = segments.sort_values("monetary", ascending=False).iloc[0]
        d = segments.sort_values("discount_share", ascending=False).iloc[0]
        tgt = s if inp.brand_positioning in ("premium", "super-premium") else (d if inp.intro_discount > 0.1 else s)
        recs.append(Recommendation("Target consumer", f"Prioritise the '{tgt['segment']}' segment for trial and CRM follow-up",
            "Chosen from measured RFM segment characteristics matched to the planned positioning/promotion.",
            f"{tgt['share_pct']:.0f}% of customers; avg frequency {tgt['frequency']:.1f}, spend {tgt['monetary']:,.0f}, discount share {tgt['discount_share']:.0%}.",
            "Focuses spend on customers with demonstrated behaviour.", "Segments describe past purchasing, not demographics or future response.", "Medium", "rule-based"))
    # 10. reliability / pilot
    hi = [f for lv, f in res["reliability"]["flags"] if lv == "high"]
    if lvl == "Low" or hi or res["error_model"]["log_ratio_sd"] > 0.5:
        recs.append(Recommendation("Risk mitigation", "Pilot in one region / channel before national rollout",
            "Forecast uncertainty and/or data-support issues are material.",
            "; ".join(hi[:2]) or f"Out-of-sample launch totals vary by x{np.exp(res['error_model']['log_ratio_sd']):.1f} (1 SD).",
            "Replaces model uncertainty with observed launch data at small cost.", "Slower rollout.", "High", "rule-based"))
    return recs
