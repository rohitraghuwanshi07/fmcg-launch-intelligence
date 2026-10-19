"""CSV and PDF report generation (reportlab). Text is sanitised to Latin-1 for the built-in PDF fonts."""
from __future__ import annotations
import io
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from . import config

def _t(x) -> str:
    s = str(x).replace("\u20b9", "INR ").replace("\u2013", "-").replace("\u2014", "-").replace("\u2265", ">=").replace("\u2264", "<=")
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _m(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}"

def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444")))
    ss.add(ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8, leading=9.5))
    ss.add(ParagraphStyle("Banner", parent=ss["BodyText"], fontSize=9, textColor=colors.white, backColor=colors.HexColor("#b45309"),
                          borderPadding=4, leading=12))
    return ss

def _table(rows, header, widths=None, ss=None):
    ss = ss or _styles()
    data = [[Paragraph(f"<b>{_t(h)}</b>", ss["Cell"]) for h in header]] + [[Paragraph(_t(c), ss["Cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")])]))
    return t

def _header(res, decision, meta, ss, title):
    inp = res["inputs"]
    el = [Paragraph(_t(title), ss["Title"]), Paragraph(_t(f"{inp['product_name']} - {inp['category']} - launch {inp['launch_date']}"), ss["Heading3"])]
    if meta.get("mode") != "real":
        el.append(Paragraph("DEMONSTRATION: models and market data are SIMULATED. Figures illustrate the workflow and are not real-world observations.", ss["Banner"]))
        el.append(Spacer(1, 6))
    el.append(Paragraph("<b>Recommendation: " + _t(decision["decision"]) + "</b>. " + _t(decision["rationale"]), ss["BodyText"]))
    el.append(Paragraph(_t(config.DISCLAIMER), ss["Small"]))
    return el

def _kpis(res, ss):
    H = res["horizons"].set_index("horizon"); ue = res["unit_economics"]["regular"]
    rows = [["Commercial hurdle attainment probability [Model estimate]", f"{res['p_hurdle']:.0%} (target profit {_m(res['inputs']['target_profit'])} after launch costs within {res['hurdle_weeks']}w)"],
            ["Expected units 1 / 3 / 6 / 12 months [Model estimate]", " / ".join(_m(H.loc[k, 'units_mean']) for k in H.index)],
            ["Expected 6-month manufacturer net revenue [Model estimate]", f"{config.CURRENCY} {_m(H.loc['6 months', 'net_revenue_mean'])}"],
            ["Expected 6-month contribution after launch costs [Model estimate]", f"{config.CURRENCY} {_m(res['expected_after_launch_costs'])}"],
            ["Unit contribution / margin [Calculated from assumptions]", f"{_m(ue['contribution'], 2)} / {ue['margin_pct']:.1%}"],
            ["Break-even units (regular economics) [Calculated]", _m(res["breakeven"]["break_even_units_regular"])],
            ["Model reliability [Validation-based]", f"{res['reliability']['level']} (score {res['reliability']['score']})"]]
    return _table(rows, ["Metric", "Value"], [8.5 * cm, 8.5 * cm], ss)

def executive_pdf(res, recs, decision, meta) -> bytes:
    ss = _styles(); buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el = _header(res, decision, meta, ss, "Launch Assessment - Executive Summary")
    el += [Spacer(1, 8), _kpis(res, ss), Spacer(1, 8), Paragraph("Scenarios (6 months)", ss["Heading3"])]
    sc = res["scenarios"]
    el.append(_table([[r.scenario, r.demand_quantile, _m(r.units_6m), _m(r.net_revenue_6m), _m(r.after_launch_costs_6m), "Yes" if r.meets_hurdle else "No"] for r in sc.itertuples()],
                     ["Scenario", "Demand level", "Units", "Net revenue", "After launch costs", "Meets hurdle"], None, ss))
    el += [Paragraph(_t("Demand level is an empirical quantile of out-of-fold launch errors; cost (+5% / 0 / -3%) and availability (0.9 / 1 / 1) adjustments are manual assumptions."), ss["Small"]),
           Paragraph("Top risks", ss["Heading3"])]
    for lv, f in res["reliability"]["flags"][:5]: el.append(Paragraph(_t(f"[{lv}] {f}"), ss["BodyText"]))
    el.append(Paragraph("Recommended next steps", ss["Heading3"]))
    for r in recs[:6]: el.append(Paragraph("<b>" + _t(r.area) + ":</b> " + _t(r.action) + " <i>(" + _t(f"{r.source}, confidence {r.confidence}") + ")</i>", ss["BodyText"]))
    doc.build(el); return buf.getvalue()

def detailed_pdf(res, recs, decision, meta, whatifs=None, metrics=None, market=None, sensitivity=None) -> bytes:
    ss = _styles(); buf = io.BytesIO(); inp = res["inputs"]
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    el = _header(res, decision, meta, ss, "Launch Assessment - Detailed Analytical Report")
    el += [Spacer(1, 6), Paragraph("1. Proposed product", ss["Heading2"]),
           _table([[k, v] for k, v in [("Name", inp["product_name"]), ("Category", inp["category"]), ("Positioning", inp["brand_positioning"]),
                   ("Pack", f"{inp['pack_size']:g} {inp['pack_unit']} ({inp['packaging_type']})"), ("Shelf price (incl. tax)", f"{config.CURRENCY} {inp['price']:.2f}"),
                   ("Target regions", ", ".join(inp["regions"])), ("Outlets", json.dumps(inp["outlets"])), ("Launch date", inp["launch_date"])]], ["Field", "Value"], [5 * cm, 12 * cm], ss),
           Paragraph("2. Headline results", ss["Heading2"]), _kpis(res, ss), Paragraph("3. Hurdle definition", ss["Heading2"]), Paragraph(_t(config.HURDLE_RULE), ss["BodyText"]),
           Paragraph("4. Unit economics (manufacturer view) [Calculated from user assumptions]", ss["Heading2"])]
    u, ip = res["unit_economics"]["regular"], res["unit_economics"]["intro"]
    el.append(_table([[k.replace("_", " "), _m(u[k], 2), _m(ip[k], 2)] for k in ("consumer_price", "price_ex_tax", "trade_margin_amt", "discount_cost", "net_revenue", "wastage", "variable_cost", "contribution")],
                     ["Per unit", "Regular", "Introductory period"], None, ss))
    el += [Paragraph("5. Demand and financial projection [Model estimate; P10-P90 from out-of-fold launch errors]", ss["Heading2"])]
    h = res["horizons"]
    el.append(_table([[r.horizon, _m(r.units_mean), f"{_m(r.units_p10)} - {_m(r.units_p90)}", _m(r.net_revenue_mean), _m(r.contribution_mean), _m(r.after_launch_costs_mean)] for r in h.itertuples()],
                     ["Horizon", "Units (mean)", "Units P10-P90", "Net revenue", "Contribution", "After launch costs"], None, ss))
    el += [Paragraph("6. Scenarios", ss["Heading2"])]
    sc = res["scenarios"]
    el.append(_table([[r.scenario, r.demand_quantile, f"x{r.demand_multiplier:.2f}", f"x{r.cost_multiplier:.2f}", f"x{r.availability_multiplier:.2f}", _m(r.units_6m), _m(r.contribution_6m), _m(r.after_launch_costs_6m), "Yes" if r.meets_hurdle else "No"] for r in sc.itertuples()],
                     ["Scenario", "Quantile", "Demand", "Cost", "Avail.", "Units 6m", "Contribution 6m", "After launch costs", "Hurdle"], None, ss))
    b, inv, wc = res["breakeven"], res["inventory"], res["working_capital"]
    el += [Paragraph("7. Break-even, inventory and cash", ss["Heading2"]),
           _table([["Break-even units (regular economics)", _m(b["break_even_units_regular"])], ["Break-even revenue (manufacturer net)", _m(b["break_even_revenue_regular"])],
                   ["Break-even week (median path)", str(b["break_even_week_base"] or "not within 12 months")], ["P(break-even within 12 months)", f"{b['p_break_even_12m']:.0%}"],
                   ["Planned vs recommended (P90) opening inventory", f"{_m(inv['planned_initial'])} vs {_m(inv['recommended_initial_p90'])}"],
                   ["P(stock-out) / expected lost units", f"{inv['p_stockout']:.0%} / {_m(inv['expected_lost_units'])}"],
                   ["Peak funding need (median path)", f"{config.CURRENCY} {_m(wc['peak_funding_need'])} (week {wc['week_of_peak']})"],
                   ["Budget utilisation", f"{_m(res['budget']['utilisation_pct'], 0)}% of total launch budget"]], ["Item", "Value"], [9 * cm, 8 * cm], ss)]
    if whatifs is not None:
        el += [Paragraph("8. What-if analysis (scenario estimates, not causal effects)", ss["Heading2"]),
               _table([[r.scenario, f"{r.p_hurdle:.0%}" if np.isfinite(r.p_hurdle) else "n/a", _m(r.expected_after_launch_costs), _m(r.units_6m)] for r in whatifs.itertuples()],
                      ["Scenario", "P(hurdle)", "Expected after launch costs", "Units 6m"], None, ss)]
    if sensitivity is not None:
        el += [Paragraph("Sensitivity (+/-10% one-at-a-time)", ss["Heading3"]),
               _table([[r.driver, _m(r.low), _m(r.high)] for r in sensitivity.itertuples()], ["Driver", "-10% change in result", "+10% change in result"], None, ss)]
    if market and market.get("available"):
        el += [Paragraph("9. Competitive position", ss["Heading2"]),
               Paragraph(_t(f"Price per 100 units is at the {market['percentile']:.0f}th percentile of {market['n_comparables']} comparables (median {market['median_ppu100']:.1f}); implied tier: {market['implied_tier']}. "
                            "Comparables are dataset-observed historical prices and/or user-provided entries - not live market data."), ss["BodyText"])]
    el += [Paragraph("10. Risks and reliability", ss["Heading2"])]
    for lv, f in res["reliability"]["flags"]: el.append(Paragraph(_t(f"[{lv}] {f}"), ss["BodyText"]))
    el += [Paragraph("11. Recommendations", ss["Heading2"]),
           _table([[r.area, r.action, r.reason, r.evidence, r.tradeoff, f"{r.source} / {r.confidence}"] for r in recs], ["Area", "Action", "Why", "Evidence", "Trade-off", "Source / confidence"],
                  [2 * cm, 3.6 * cm, 3.2 * cm, 3.6 * cm, 2.8 * cm, 2.3 * cm], ss), PageBreak(),
           Paragraph("12. Data sources, freshness and model evaluation", ss["Heading2"]),
           Paragraph(_t(f"Dataset: {meta.get('name', 'unknown')} (mode: {meta.get('mode', 'unknown')}), {meta.get('start')} to {meta.get('end')}. "
                        f"Launch-outcome labels: {'yes' if meta.get('has_launch_labels') else 'NO - a launch-success classifier is not trained; hurdle probability is a scenario estimate.'}"), ss["BodyText"])]
    if metrics and metrics.get("coldstart"):
        t = metrics["coldstart"]["metrics"]["test"]
        el.append(_table([[k, f"{v['WAPE']:.3f}", f"{v['MAE']:.1f}", f"{v['product_MdAPE_26w']:.2f}"] for k, v in t.items()], ["Cold-start model (held-out products)", "WAPE", "MAE", "Product MdAPE 26w"], None, ss))
    em = res["error_model"]
    el += [Paragraph(_t(f"Uncertainty: {em['n_products']} out-of-fold launches; SD of log(actual/forecast) of 26-week totals = {em['log_ratio_sd']:.2f}; median ratio x{np.exp(em['log_ratio_median']):.2f}."), ss["BodyText"]),
           Paragraph("13. Assumptions requiring validation", ss["Heading2"])]
    for a in [f"Marketing response: max uplift {inp['max_uplift']:.0%}, half-saturation {inp['half_sat_per_outlet']:,.0f}/outlet, post-campaign retention {inp['post_campaign_retention']:.0%} (assumption; no campaign data).",
              f"Price response: within-product elasticity {res['adjustments']['elasticity']:.2f} (observational).", f"Cross-price elasticity {inp['cross_elasticity']} (assumption); competitor price change {inp['competitor_price_change_pct']}%.",
              f"Tax {inp['tax_rate']:.0%}, trade margin {inp['trade_margin']:.0%}, wastage {inp['wastage_pct']:.0%}, receivable days {inp['receivable_days']}.", "Unit cost realisation uncertainty SD 3% (assumption).",
              "Scenario cost (+5%/0/-3%) and availability (0.9/1/1) multipliers are manual.", "Per-outlet demand is taken from sampled historical stores; planned outlets are assumed comparable."]:
        el.append(Paragraph(_t("- " + a), ss["BodyText"]))
    doc.build(el); return buf.getvalue()

def results_csvs(res, recs=None, whatifs=None) -> dict[str, bytes]:
    out = {"horizons.csv": res["horizons"].to_csv(index=False).encode(), "scenarios.csv": res["scenarios"].to_csv(index=False).encode(),
           "weekly_forecast.csv": res["weekly"].to_csv(index=False).encode()}
    if recs: out["recommendations.csv"] = pd.DataFrame([r.to_dict() for r in recs]).to_csv(index=False).encode()
    if whatifs is not None: out["what_if.csv"] = whatifs.to_csv(index=False).encode()
    return out
