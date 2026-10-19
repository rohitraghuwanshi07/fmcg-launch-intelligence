"""Pilot-region selection as a multi-armed bandit (Thompson sampling vs epsilon-greedy vs uniform).

Question answered: if we pilot in several regions before a national roll-out, how should pilot outlets be
allocated between regions so that we both LEARN which region is strongest and LOSE LITTLE by piloting in weak ones?

Setting (SIMULATION, labelled as such): the cold-start model gives a prior mean demand per outlet-week for each
region; the true regional level deviates from that prior by a log-normal "surprise" (ASSUMPTION, default sd 0.25);
each outlet-round yields a noisy observation. Posterior on the log-level is Gaussian (conjugate, known noise).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from . import scenario_engine as se
from dataclasses import replace


def region_priors(inp: se.LaunchInputs, ctx: dict) -> pd.DataFrame:
    rows = []
    for r in ctx["coldstart"]["support"]["regions"]:
        i = replace(inp, regions=[r])
        per = se.predict_per_outlet(i, ctx)
        wk = per.iloc[:13]
        tot = sum(wk[c].mean() * n for c, n in inp.outlets.items() if c in wk.columns and n > 0)
        n_out = max(sum(inp.outlets.values()), 1)
        rows.append(dict(region=r, prior_units_per_outlet_week=tot / n_out))
    return pd.DataFrame(rows)


def simulate_policies(priors: np.ndarray, surprise_sd: float = 0.25, obs_sd: float = 0.5, rounds: int = 12,
                      batch: int = 20, n_runs: int = 400, seed: int = config.SEED, eps: float = 0.1) -> dict:
    rng = np.random.default_rng(seed)
    K = len(priors)
    mu0 = np.log(np.maximum(priors, 1e-6))
    tau2 = surprise_sd ** 2
    true = mu0[None, :] + rng.normal(0, surprise_sd, (n_runs, K))          # latent log-levels
    best = true.argmax(1)
    out = {}
    for policy in ("Uniform", "Epsilon-greedy", "Thompson sampling"):
        prec = np.full((n_runs, K), 1 / tau2)
        sm = np.tile(mu0 / tau2, (n_runs, 1))                                # precision-weighted sum
        regret = np.zeros((n_runs, rounds))
        for t in range(rounds):
            post_mu, post_var = sm / prec, 1 / prec
            if policy == "Uniform":
                share = np.full((n_runs, K), 1 / K)
            elif policy == "Epsilon-greedy":
                share = np.full((n_runs, K), eps / K)
                share[np.arange(n_runs), post_mu.argmax(1)] += 1 - eps
            else:   # batch Thompson: each outlet in the batch samples a posterior draw and picks the argmax
                draws = rng.normal(post_mu[:, None, :], np.sqrt(post_var)[:, None, :], (n_runs, batch, K))
                pick = draws.argmax(2)
                share = np.stack([(pick == k).mean(1) for k in range(K)], axis=1)
            n_k = share * batch
            obs = true + rng.normal(0, obs_sd, (n_runs, K)) / np.sqrt(np.maximum(n_k, 1e-9))   # mean of n_k obs
            upd = n_k > 1e-9
            prec = prec + np.where(upd, n_k / obs_sd ** 2, 0)
            sm = sm + np.where(upd, n_k * obs / obs_sd ** 2, 0)
            lvl = np.exp(true)
            regret[:, t] = (lvl.max(1) - (share * lvl).sum(1)) / lvl.max(1)   # relative per-outlet-week shortfall
        final_pick = (sm / prec).argmax(1)
        out[policy] = dict(cum_regret=np.cumsum(regret, axis=1).mean(0), p_identify_best=float((final_pick == best).mean()),
                           mean_final_share_best=float(regret[:, -1].mean()))
    return out


def posterior_best_probability(priors: np.ndarray, surprise_sd: float = 0.25, n: int = 20000, seed: int = config.SEED) -> np.ndarray:
    """Prior probability that each region is the strongest (before any pilot data)."""
    rng = np.random.default_rng(seed)
    mu0 = np.log(np.maximum(priors, 1e-6))
    d = mu0[None, :] + rng.normal(0, surprise_sd, (n, len(priors)))
    return np.bincount(d.argmax(1), minlength=len(priors)) / n
