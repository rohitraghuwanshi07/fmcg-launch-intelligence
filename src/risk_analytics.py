"""Risk analytics on the Monte-Carlo launch simulation.

* tail_risk      - VaR / CVaR (expected shortfall) of profit after launch costs, P(loss), downside deviation
* sobol_indices  - variance-based global sensitivity (Saltelli/Jansen estimators, quasi-random Sobol sequence)
                   -> unlike a one-at-a-time tornado it captures interactions between drivers.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.stats import qmc
from . import config
from . import scenario_engine as se


def draw_noise(ctx: dict, n: int, seed: int = config.SEED) -> dict:
    """Common random numbers: reuse the same draws when comparing strategies (variance reduction)."""
    cs = ctx["coldstart"]
    rng = np.random.default_rng(seed)
    W = config.COLD_START_WEEKS
    m = np.exp(se._log_ratio_draws(cs, n, rng))
    sd = cs["weekly_noise_sd"]
    eps = rng.normal(-sd ** 2 / 2, sd, (n, W))
    cm = np.clip(rng.normal(1.0, 0.03, n), 0.8, 1.3)
    return dict(m=m, eps=eps, cm=cm)


def simulate_profit(inp: se.LaunchInputs, ctx: dict, noise: dict) -> np.ndarray:
    """Profit after ALL launch spend at the hurdle horizon, one value per simulated world."""
    base_D, _, _ = se.weekly_base_demand(inp, ctx, config.COLD_START_WEEKS)
    D = base_D[None, :] * noise["m"][:, None] * np.exp(noise["eps"])
    S = se._simulate(D, inp, noise["cm"])
    hw = min(inp.hurdle_weeks, config.COLD_START_WEEKS)
    return S["cum_after_fixed"][:, hw - 1]


def tail_risk(profit: np.ndarray, target: float = 0.0, alpha: float = 0.05) -> dict:
    p = np.sort(np.asarray(profit, float))
    k = max(int(np.ceil(alpha * len(p))), 1)
    var = float(np.percentile(p, 100 * alpha))
    cvar = float(p[:k].mean())
    down = np.minimum(p - target, 0.0)
    return dict(mean=float(p.mean()), median=float(np.median(p)), std=float(p.std()), var=var, cvar=cvar,
                p_loss=float((p < 0).mean()), p_hurdle=float((p >= target).mean()),
                downside_dev=float(np.sqrt(np.mean(down ** 2))), p90=float(np.percentile(p, 90)),
                risk_reward=float(p.mean() / p.std()) if p.std() > 0 else float("nan"), alpha=alpha)


def risk_adjusted_value(profit: np.ndarray, lam: float = 0.5, alpha: float = 0.05) -> float:
    """(1 - lam) * E[profit] + lam * CVaR_alpha[profit]; lam = 0 is risk-neutral, 1 is fully risk-averse."""
    t = tail_risk(profit, alpha=alpha)
    return (1 - lam) * t["mean"] + lam * t["cvar"]


SOBOL_FACTORS = {  # name -> (low, high, how to apply)
    "Shelf price": (-12.0, 12.0),
    "Manufacturing cost": (-12.0, 12.0),
    "Marketing budget": (-40.0, 40.0),
    "Distribution (outlets x)": (0.6, 1.6),
    "Demand level": (-25.0, 25.0),
    "Shelf availability": (0.80, 1.00),
}


def _apply(inp: se.LaunchInputs, x: np.ndarray) -> se.LaunchInputs:
    i = se.apply_whatif(inp, price_pct=x[0], mfg_cost_pct=x[1], marketing_pct=x[2], outlets_mult=x[3], demand_pct=x[4])
    return replace(i, availability=float(x[5]))


def sobol_indices(inp: se.LaunchInputs, ctx: dict, n: int = 32, seed: int = config.SEED) -> pd.DataFrame:
    """First-order (S1) and total-order (ST) Sobol indices of expected profit after launch costs.
    Cost: n * (k + 2) deterministic model evaluations (n must be a power of two for the Sobol sequence)."""
    names = list(SOBOL_FACTORS)
    k = len(names)
    lo = np.array([SOBOL_FACTORS[c][0] for c in names]); hi = np.array([SOBOL_FACTORS[c][1] for c in names])
    sam = qmc.Sobol(d=2 * k, scramble=True, seed=seed).random(n)
    A = lo + sam[:, :k] * (hi - lo)
    B = lo + sam[:, k:] * (hi - lo)

    def f(X):
        return np.array([se.run_assessment(_apply(inp, x), ctx, mc=False)["expected_after_launch_costs"] for x in X])
    fA, fB = f(A), f(B)
    var = np.var(np.concatenate([fA, fB]))
    rows = []
    for i, name in enumerate(names):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fAB = f(ABi)
        s1 = np.mean(fB * (fAB - fA)) / var if var > 0 else 0.0           # Saltelli 2010
        st = 0.5 * np.mean((fA - fAB) ** 2) / var if var > 0 else 0.0      # Jansen 1999
        rows.append(dict(driver=name, S1=float(np.clip(s1, 0, 1)), ST=float(np.clip(st, 0, 1.2))))
    out = pd.DataFrame(rows)
    out["interaction"] = (out.ST - out.S1).clip(lower=0)
    return out.sort_values("ST", ascending=False).reset_index(drop=True)
