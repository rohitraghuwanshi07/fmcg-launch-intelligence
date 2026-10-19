"""Red Team: reverse stress testing - "what would have to go wrong for this launch to fail?"

Instead of asking how the plan performs under chosen scenarios, we search for the SMALLEST (most plausible)
combination of adverse shocks that pushes P(meets hurdle) below a threshold:

    minimise   z(s) = sqrt( sum_i (s_i / sigma_i)^2 )      "distance to failure" in plausibility units
    subject to P(hurdle | plan shocked by s) <= threshold

* single-factor breakpoints are found by bisection (P(hurdle) is monotone in each adverse shock)
* the joint minimum-norm shock is found by differential evolution (non-smooth, simulation-based constraint),
  and is never worse than the best single-factor breakpoint (that point is a feasible candidate)
* sigma_i are PLAUSIBILITY scales (assumptions, except demand which uses the model's own out-of-fold error)
The shocks are applied ON TOP of the uncertainty the Monte-Carlo already simulates (systematic misses, not noise).
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from . import config
from . import scenario_engine as se
from . import risk_analytics as ra

FACTORS = {  # key: (label, unit, max adverse magnitude searched, plausibility sigma [None = from model])
    "demand_pct": ("Demand shortfall vs forecast", "%", 70.0, None),
    "mfg_cost_pct": ("Manufacturing cost inflation", "%", 60.0, 6.0),
    "price_pct": ("Forced price cut", "%", 20.0, 5.0),
    "outlets_pct": ("Distribution shortfall", "%", 60.0, 15.0),
    "delay_weeks": ("Launch delay", "weeks", 12.0, 3.0),
    "competitor_price_pct": ("Competitor price cut", "%", 25.0, 5.0),
}
KEYS = list(FACTORS)


def sigmas(ctx: dict) -> dict:
    sd = float(np.std(ctx["coldstart"]["log_ratio"]))
    return {k: (sd if v[3] is None else v[3]) for k, v in FACTORS.items()}


def apply_shock(inp: se.LaunchInputs, s: dict) -> se.LaunchInputs:
    kw = {}
    if s.get("demand_pct", 0) > 0: kw["demand_pct"] = -s["demand_pct"]
    if s.get("mfg_cost_pct", 0) > 0: kw["mfg_cost_pct"] = s["mfg_cost_pct"]
    if s.get("price_pct", 0) > 0: kw["price_pct"] = -s["price_pct"]
    if s.get("outlets_pct", 0) > 0: kw["outlets_mult"] = 1 - s["outlets_pct"] / 100
    if s.get("delay_weeks", 0) >= 0.5: kw["delay_weeks"] = int(round(s["delay_weeks"]))
    if s.get("competitor_price_pct", 0) > 0: kw["competitor_price_pct"] = -s["competitor_price_pct"]
    return se.apply_whatif(inp, **kw) if kw else inp


def zdist(s: dict, sig: dict) -> float:
    """Plausibility distance. Demand is measured in log space against the model's own forecast-error SD."""
    t = 0.0
    for k, v in s.items():
        if v <= 0: continue
        x = abs(np.log(max(1 - v / 100, 1e-6))) if k == "demand_pct" else v
        t += (x / sig[k]) ** 2
    return float(np.sqrt(t))


def _p(inp, ctx, noise, s) -> float:
    i = apply_shock(inp, s)
    return float((ra.simulate_profit(i, ctx, noise) >= inp.target_profit).mean())


def breakpoints(inp: se.LaunchInputs, ctx: dict, threshold: float = 0.5, n_sims: int = 800, seed: int = config.SEED) -> pd.DataFrame:
    """Bisection per factor: the adverse shock at which P(hurdle) first drops to <= threshold."""
    noise = ra.draw_noise(ctx, n_sims, seed); sig = sigmas(ctx); rows = []
    for k, (lab, unit, hi, _) in FACTORS.items():
        lo_, hi_ = 0.0, hi
        if _p(inp, ctx, noise, {k: hi_}) > threshold:
            rows.append(dict(factor=lab, key=k, breakpoint=np.nan, unit=unit, z=np.nan, reached=False, searched_up_to=hi)); continue
        for _ in range(14):
            mid = (lo_ + hi_) / 2
            if _p(inp, ctx, noise, {k: mid}) > threshold: lo_ = mid
            else: hi_ = mid
        bp = hi_
        rows.append(dict(factor=lab, key=k, breakpoint=bp, unit=unit, z=zdist({k: bp}, sig), reached=True, searched_up_to=hi))
    return pd.DataFrame(rows)


def reverse_stress_test(inp: se.LaunchInputs, ctx: dict, threshold: float = 0.5, n_sims: int = 500, maxiter: int = 15,
                        seed: int = config.SEED) -> dict:
    noise = ra.draw_noise(ctx, n_sims, seed); sig = sigmas(ctx)
    base_p = _p(inp, ctx, noise, {})
    bp = breakpoints(inp, ctx, threshold, n_sims, seed)
    bounds = [(0.0, FACTORS[k][2]) for k in KEYS]

    def obj(x):
        s = dict(zip(KEYS, x))
        p = _p(inp, ctx, noise, s)
        return zdist(s, sig) + (200.0 * (p - threshold) + 5.0 if p > threshold else 0.0)
    r = differential_evolution(obj, bounds, seed=seed, maxiter=maxiter, popsize=6, tol=1e-3, polish=False, init="sobol", updating="immediate")
    cand = [dict(zip(KEYS, r.x))]
    for _, row in bp[bp.reached].iterrows():
        cand.append({row.key: float(row.breakpoint)})              # single-factor breakpoints are always feasible candidates
    feas = [(zdist(c, sig), c, _p(inp, ctx, noise, c)) for c in cand]
    feas = [f for f in feas if f[2] <= threshold + 0.02]
    if not feas:
        return dict(found=False, base_p=base_p, breakpoints=bp, threshold=threshold, sigmas=sig)
    z, best, p_at = min(feas, key=lambda t: t[0])
    best = {k: float(v) for k, v in best.items() if v > 1e-6}
    contrib = {k: (abs(np.log(max(1 - v / 100, 1e-6))) / sig[k] if k == "demand_pct" else v / sig[k]) ** 2 for k, v in best.items()}
    tot = sum(contrib.values()) or 1.0
    shares = {k: c / tot for k, c in contrib.items()}
    label = "Fragile" if z < 1.0 else "Moderately robust" if z < 2.0 else "Robust"
    return dict(found=True, shock=best, z=float(z), p_at_failure=float(p_at), base_p=base_p, threshold=threshold, shares=shares,
                verdict=label, breakpoints=bp, sigmas=sig, n_evals=int(r.nfev))


def pre_mortem(res: dict, inp: se.LaunchInputs) -> str:
    """One-paragraph failure story from the most plausible failing combination."""
    if not res.get("found"):
        return "No combination of the tested shocks within plausible ranges brings the launch below the failure threshold - the plan is robust to these risks."
    parts = []
    for k, v in sorted(res["shock"].items(), key=lambda kv: -res["shares"][kv[0]]):
        lab, unit, *_ = FACTORS[k]
        parts.append(f"{lab.lower()} of {v:.0f}{'' if unit == 'weeks' else unit}{' weeks' if unit == 'weeks' else ''}")
    return (f"It is six months after launch and '{inp.product_name}' missed its hurdle. The cheapest explanation in plausibility terms "
            f"({res['z']:.1f} sigma from plan) is: " + "; ".join(parts) + ". "
            f"{'That is within one standard deviation of normal planning error - treat this launch as fragile.' if res['z'] < 1 else 'That requires a clearly worse-than-planned world, so the plan has real headroom.' if res['z'] >= 2 else 'It needs a worse-than-usual but not extreme outcome.'}")
