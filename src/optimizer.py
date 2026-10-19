"""Launch-strategy optimisation.

1. optimize_strategy  - global stochastic optimisation (differential evolution) of price, marketing scale,
                        introductory discount and opening inventory. Objective is a RISK-ADJUSTED value
                        (1-lambda)*E[profit] + lambda*CVaR5[profit], evaluated with common random numbers so
                        candidate strategies are compared on the same simulated worlds.
2. allocate_budget_dp - exact dynamic programme (multiple-choice knapsack) splitting a budget across channels
                        with diminishing-returns response curves; allocate_greedy is the O(B log C) heap version
                        (optimal for concave curves) and is used as a cross-check in the tests.
Channel efficacy parameters are ASSUMPTIONS (no campaign data in the demo) and are shown as such.
"""
from __future__ import annotations
import heapq
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from . import config
from . import scenario_engine as se
from . import risk_analytics as ra

VARS = ["Shelf price (x plan)", "Marketing budget (x plan)", "Intro discount", "Opening inventory (x plan)"]


def _bounds(ctx: dict, inp: se.LaunchInputs):
    lo_r, hi_r = ctx["ratio_range"]           # price ratios supported by the elasticity evidence
    return [(max(0.85, lo_r), min(1.15, hi_r)), (0.5, 1.5), (0.0, 0.20), (0.5, 2.0)]


def apply_strategy(inp: se.LaunchInputs, x) -> se.LaunchInputs:
    price_x, mkt_x, disc, inv_x = x
    i = se.apply_whatif(inp, price_pct=(price_x - 1) * 100, marketing_pct=(mkt_x - 1) * 100, discount=float(disc))
    return replace(i, initial_inventory=float(inp.initial_inventory * inv_x))


def optimize_strategy(inp: se.LaunchInputs, ctx: dict, lam: float = 0.5, n_sims: int = 500, maxiter: int = 18,
                      popsize: int = 6, budget_cap: float | None = None, seed: int = config.SEED) -> dict:
    noise = ra.draw_noise(ctx, n_sims, seed)
    cap = budget_cap if budget_cap else None
    history: list[float] = []

    def score(x):
        cand = apply_strategy(inp, x)
        spend = cand.fixed_costs()
        pen = 0.0
        if cap and spend > cap:
            pen = 5.0 * (spend - cap)           # soft budget constraint
        return -(ra.risk_adjusted_value(ra.simulate_profit(cand, ctx, noise), lam) - pen)

    base_x = np.array([1.0, 1.0, inp.intro_discount, 1.0])
    res = differential_evolution(score, _bounds(ctx, inp), seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-4,
                                 polish=False, init="sobol", updating="immediate", x0=base_x,
                                 callback=lambda xk, convergence=None: history.append(-score(xk)))
    best_x = res.x
    best_inp = apply_strategy(inp, best_x)

    def summarise(i):
        p = ra.simulate_profit(i, ctx, noise)
        t = ra.tail_risk(p, i.target_profit)
        return dict(expected_profit=t["mean"], cvar5=t["cvar"], p_hurdle=t["p_hurdle"], p_loss=t["p_loss"],
                    risk_adjusted=ra.risk_adjusted_value(p, lam), price=i.price, marketing_spend=sum(getattr(i, k) for k in se.MARKETING_KEYS),
                    intro_discount=i.intro_discount, initial_inventory=i.initial_inventory, fixed_costs=i.fixed_costs())
    cur, opt = summarise(inp), summarise(best_inp)
    comp = pd.DataFrame({"Current plan": cur, "Optimised": opt}).T
    bnds = _bounds(ctx, inp)
    at_bound = [v for v, xv, (lo, hi) in zip(VARS, best_x, bnds) if min(xv - lo, hi - xv) < 0.02 * (hi - lo)]
    return dict(best_inputs=best_inp, current=cur, optimised=opt, comparison=comp, x=dict(zip(VARS, map(float, best_x))),
                history=history, n_evals=int(res.nfev), lam=lam, converged=bool(res.success), at_bound=at_bound,
                bounds=dict(zip(VARS, bnds)))


# ----------------------------------------------------------------------------- budget allocation
CHANNEL_PRIORS = {  # ASSUMPTION: relative efficacy weight and saturation scale (spend where ~63 % of max effect is reached)
    "ad_budget": (1.00, 700_000.0),
    "digital_budget": (1.30, 450_000.0),
    "retail_activation_budget": (1.10, 300_000.0),
    "influencer_budget": (0.80, 180_000.0),
}


def response(x: float, weight: float, scale: float) -> float:
    return weight * (1.0 - np.exp(-x / scale))


def allocate_budget_dp(total: float, priors: dict = CHANNEL_PRIORS, step: float = 25_000.0) -> dict:
    """Exact DP over budget units: f[c][b] = max_{x<=b} f[c-1][b-x] + response_c(x). O(C * B^2)."""
    names = list(priors)
    B = int(round(total / step))
    f = np.zeros((len(names) + 1, B + 1))
    choice = np.zeros((len(names), B + 1), dtype=int)
    for c, n in enumerate(names, start=1):
        w, s = priors[n]
        gains = response(np.arange(B + 1) * step, w, s)
        for b in range(B + 1):
            cand = f[c - 1][b - np.arange(b + 1)] + gains[: b + 1]
            j = int(np.argmax(cand))
            f[c][b], choice[c - 1][b] = cand[j], j
    alloc, b = {}, B
    for c in range(len(names), 0, -1):
        x = choice[c - 1][b]
        alloc[names[c - 1]] = float(x * step)
        b -= x
    return dict(allocation=alloc, value=float(f[len(names)][B]), step=step)


def allocate_greedy(total: float, priors: dict = CHANNEL_PRIORS, step: float = 25_000.0) -> dict:
    """Max-heap on marginal gain; optimal because every response curve is concave."""
    names = list(priors)
    B = int(round(total / step))
    alloc = {n: 0 for n in names}
    def gain(n):
        w, s = priors[n]; k = alloc[n]
        return response((k + 1) * step, w, s) - response(k * step, w, s)
    heap = [(-gain(n), n) for n in names]
    heapq.heapify(heap)
    for _ in range(B):
        g, n = heapq.heappop(heap)
        alloc[n] += 1
        heapq.heappush(heap, (-gain(n), n))
    out = {n: k * step for n, k in alloc.items()}
    return dict(allocation=out, value=float(sum(response(out[n], *priors[n]) for n in names)), step=step)


def compare_allocations(inp: se.LaunchInputs, priors: dict = CHANNEL_PRIORS, step: float = 25_000.0) -> pd.DataFrame:
    total = sum(getattr(inp, k) for k in se.MARKETING_KEYS)
    cur = {k: getattr(inp, k) for k in priors}
    opt = allocate_budget_dp(total, priors, step)["allocation"]
    val = lambda a: float(sum(response(a[n], *priors[n]) for n in priors))
    df = pd.DataFrame({"current": cur, "optimal": opt})
    df["shift"] = df.optimal - df.current
    df.attrs.update(current_value=val(cur), optimal_value=val(opt), total=total)
    return df
