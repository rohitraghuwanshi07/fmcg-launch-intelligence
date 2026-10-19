"""Distribution-free uncertainty: split-conformal prediction on out-of-fold launch errors.

The cold-start model's product-level error log(actual / forecast) is measured on UNSEEN products
(grouped CV). Conformal calibration turns those residuals into an interval with a finite-sample
coverage guarantee: for exchangeable products, P(actual in interval) >= 1 - alpha.
"""
from __future__ import annotations
import math
import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        raise ValueError("No calibration scores.")
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[min(k, n) - 1]) if k <= n else float("inf")


def _order_stat_bounds(lr: np.ndarray, alpha: float) -> tuple[float, float]:
    """Two-sided distribution-free prediction interval from order statistics.
    Lower = floor((n+1)*alpha/2)-th smallest, upper = ceil((n+1)*(1-alpha/2))-th smallest; for exchangeable
    data this covers a new point with probability >= 1 - alpha (no centre estimated from the same data)."""
    s = np.sort(np.asarray(lr, float)); n = len(s)
    kl = math.floor((n + 1) * alpha / 2)
    ku = math.ceil((n + 1) * (1 - alpha / 2))
    lo = float(s[kl - 1]) if kl >= 1 else -np.inf
    hi = float(s[ku - 1]) if ku <= n else np.inf
    return lo, hi


def conformal_interval(point_total: float, log_ratios, alpha: float = 0.2) -> dict:
    """Interval for a 26-week total: point forecast x exp(order-statistic bounds of out-of-fold log-ratios)."""
    lr = np.asarray(log_ratios, float)
    lo_l, hi_l = _order_stat_bounds(lr, alpha)
    finite = bool(np.isfinite(lo_l) and np.isfinite(hi_l))
    return dict(alpha=alpha, centre_multiplier=math.exp(float(np.median(lr))), low_multiplier=math.exp(lo_l) if np.isfinite(lo_l) else 0.0,
                high_multiplier=math.exp(hi_l) if np.isfinite(hi_l) else float("inf"),
                low=point_total * math.exp(lo_l) if np.isfinite(lo_l) else 0.0,
                high=point_total * math.exp(hi_l) if np.isfinite(hi_l) else float("inf"),
                n_calibration=int(len(lr)), finite=finite)


def coverage_experiment(log_ratios, alpha: float = 0.2, n_rep: int = 400, seed: int = 0) -> dict:
    """Repeated random calibration/test splits of the residuals -> empirical coverage (should be ~>= 1 - alpha)."""
    lr = np.asarray(log_ratios, float)
    rng = np.random.default_rng(seed)
    n = len(lr)
    if n < 10:
        return dict(coverage=float("nan"), target=1 - alpha, n_rep=0)
    cov = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        cal, test = lr[idx[: n // 2]], lr[idx[n // 2:]]
        lo, hi = _order_stat_bounds(cal, alpha)
        cov.append(np.mean((test >= lo) & (test <= hi)))
    return dict(coverage=float(np.mean(cov)), target=1 - alpha, n_rep=n_rep, se=float(np.std(cov) / math.sqrt(n_rep)))
