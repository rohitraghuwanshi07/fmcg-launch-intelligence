"""AI Analyst: grounded explanations of a launch assessment.

Design (retrieval-grounded generation, no hallucinated numbers):
  1. build_facts()      - flatten the structured result into a small dict of verified numbers/strings
  2. offline path       - deterministic rule-based brief + keyword intent router (always available, no network)
  3. LLM path (optional)- if ANTHROPIC_API_KEY is set, the facts are sent as the ONLY allowed source; the model is
                          told to cite only those numbers
  4. grounding_check()  - every number in the LLM answer must match a fact (within rounding) or it is flagged
"""
from __future__ import annotations
import json
import os
import re
import urllib.request
import numpy as np
from . import config

LLM_MODEL = os.environ.get("FMCG_LLM_MODEL", "claude-sonnet-5-5")


def build_facts(res: dict, dec: dict, recs: list, readiness: dict | None = None, risk: dict | None = None) -> dict:
    H = res["horizons"].set_index("horizon")
    ue = res["unit_economics"]["regular"]; inp = res["inputs"]; be = res["breakeven"]
    f = {
        "product": inp["product_name"], "category": inp["category"], "currency": config.CURRENCY,
        "decision": dec["decision"], "decision_rationale": dec["rationale"],
        "p_hurdle_pct": round(100 * res["p_hurdle"], 1), "hurdle_weeks": res["hurdle_weeks"],
        "units_6m": round(float(H.loc["6 months", "units_mean"])), "net_revenue_6m": round(float(H.loc["6 months", "net_revenue_mean"])),
        "profit_after_launch_costs_6m": round(res["expected_after_launch_costs"]),
        "unit_contribution": round(ue["contribution"], 2), "unit_margin_pct": round(100 * ue["margin_pct"], 1),
        "target_margin_pct": round(100 * inp["target_margin"], 1), "shelf_price": inp["price"],
        "break_even_units": None if not np.isfinite(be["break_even_units_regular"]) else round(be["break_even_units_regular"]),
        "break_even_week": be["break_even_week_base"], "p_break_even_12m_pct": round(100 * be["p_break_even_12m"], 1),
        "p_stockout_pct": round(100 * res["inventory"]["p_stockout"], 1), "p_overstock_pct": round(100 * res["inventory"]["p_overstock"], 1),
        "model_reliability": res["reliability"]["level"], "forecast_error_sd_log": round(res["error_model"]["log_ratio_sd"], 2),
        "n_calibration_launches": res["error_model"]["n_products"],
        "recommendations": [f"{r.area}: {r.action}" for r in recs[:5]],
    }
    if readiness:
        f["readiness_index"] = round(readiness["index"], 1); f["readiness_grade"] = readiness["grade"]
        f["weakest_dimension"] = min(readiness["scores"], key=readiness["scores"].get)
        f["strongest_dimension"] = max(readiness["scores"], key=readiness["scores"].get)
    if risk:
        f.update({k: round(v, 2) if isinstance(v, float) else v for k, v in risk.items()})
    return f


def brief(facts: dict) -> str:
    c = facts["currency"]
    s = [f"**{facts['product']}** ({facts['category']}) - suggested decision: **{facts['decision']}**. {facts['decision_rationale']}",
         f"The simulation gives a **{facts['p_hurdle_pct']}%** chance of clearing the commercial hurdle within {facts['hurdle_weeks']} weeks, "
         f"with expected profit after launch costs of **{c} {facts['profit_after_launch_costs_6m']:,}** over six months on about "
         f"{facts['units_6m']:,} units.",
         f"Unit margin is {facts['unit_margin_pct']}% against a {facts['target_margin_pct']}% target "
         f"({c} {facts['unit_contribution']} contribution per unit)."]
    if facts.get("break_even_units") is not None:
        s.append(f"Break-even needs about {facts['break_even_units']:,} units; the median path reaches it in week {facts['break_even_week'] or '>52'}, "
                 f"and the chance of breaking even within 12 months is {facts['p_break_even_12m_pct']}%.")
    if "cvar5" in facts:
        s.append(f"In the worst 5% of simulated worlds the average outcome is {c} {facts['cvar5']:,.0f} (CVaR 5%); P(loss) is {100 * facts['p_loss']:.1f}%.")
    if "weakest_dimension" in facts:
        s.append(f"Readiness index {facts['readiness_index']}/100 (grade {facts['readiness_grade']}); weakest area: **{facts['weakest_dimension']}**, "
                 f"strongest: {facts['strongest_dimension']}.")
    s.append(f"Model reliability is **{facts['model_reliability']}** (forecast-error SD {facts['forecast_error_sd_log']} in log terms, "
             f"calibrated on {facts['n_calibration_launches']} unseen launches).")
    return "\n\n".join(s)


INTENTS = {
    "risk": ("risk", "loss", "worst", "downside", "cvar", "safe"), "price": ("price", "pricing", "margin", "cost"),
    "supply": ("stock", "inventory", "supply", "overstock"), "break": ("break", "payback", "recover"),
    "why": ("why", "reason", "decision", "launch"), "next": ("next", "improve", "should", "recommend", "do"),
}


def offline_answer(facts: dict, question: str) -> str:
    q = question.lower(); c = facts["currency"]
    hit = next((k for k, ws in INTENTS.items() if any(w in q for w in ws)), "why")
    if hit == "risk":
        base = f"P(meets hurdle) is {facts['p_hurdle_pct']}% and model reliability is {facts['model_reliability']}."
        if "cvar5" in facts: base += f" The worst-5% average outcome is {c} {facts['cvar5']:,.0f}; probability of a loss is {100 * facts['p_loss']:.1f}%."
        return base
    if hit == "price":
        return (f"Shelf price {c} {facts['shelf_price']} gives {c} {facts['unit_contribution']} contribution per unit "
                f"({facts['unit_margin_pct']}% vs {facts['target_margin_pct']}% target).")
    if hit == "supply":
        return f"Stock-out probability is {facts['p_stockout_pct']}% and overstock probability {facts['p_overstock_pct']}% over the horizon."
    if hit == "break":
        return (f"Break-even needs about {facts['break_even_units']:,} units; median-path week {facts['break_even_week'] or '>52'}; "
                f"P(break-even within 12 months) = {facts['p_break_even_12m_pct']}%.") if facts.get("break_even_units") is not None else \
            "Contribution per unit is not positive, so break-even is unreachable at this price and cost."
    if hit == "next":
        return "Top recommendations:\n" + "\n".join(f"- {r}" for r in facts["recommendations"])
    return f"{facts['decision']}: {facts['decision_rationale']}"


SYSTEM = ("You are a launch-analytics assistant. Answer ONLY from the JSON facts provided. Never invent or estimate numbers; "
          "if a figure is not in the facts, say it is not available. Keep it under 150 words, plain language, one clear recommendation.")


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask_llm(facts: dict, question: str, timeout: int = 30) -> str:
    body = json.dumps({"model": LLM_MODEL, "max_tokens": 500, "system": SYSTEM,
                       "messages": [{"role": "user", "content": f"FACTS:\n{json.dumps(facts, default=str)}\n\nQUESTION: {question}"}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _flatten(x):
    if isinstance(x, dict):
        for v in x.values(): yield from _flatten(v)
    elif isinstance(x, (list, tuple)):
        for v in x: yield from _flatten(v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield float(x)
    elif isinstance(x, str):
        for m in _NUM.findall(x):
            try: yield float(m.replace(",", ""))
            except ValueError: pass


def grounding_check(text: str, facts: dict, rel_tol: float = 0.02) -> dict:
    """Flag numbers in `text` that do not match any fact (after rounding / percent conversion)."""
    allowed = list(_flatten(facts))
    allowed += [a * 100 for a in allowed if abs(a) <= 1.5] + [a / 100 for a in allowed]
    bad = []
    for m in _NUM.findall(text):
        try: v = float(m.replace(",", ""))
        except ValueError: continue
        if abs(v) < 10 and float(v).is_integer(): continue   # tiny integers (5%, 12 months) are boilerplate, not claims
        if not any(abs(v - a) <= max(rel_tol * abs(a), 0.51) for a in allowed):
            bad.append(m)
    return dict(grounded=not bad, ungrounded_numbers=bad)


def answer(facts: dict, question: str) -> dict:
    if llm_available():
        try:
            txt = ask_llm(facts, question)
            return dict(text=txt, source="LLM (grounded)", check=grounding_check(txt, facts))
        except Exception as e:  # network/key problems must never break the page
            txt = offline_answer(facts, question)
            return dict(text=txt, source=f"offline rules (LLM unavailable: {type(e).__name__})", check=grounding_check(txt, facts))
    txt = offline_answer(facts, question)
    return dict(text=txt, source="offline rules", check=grounding_check(txt, facts))
