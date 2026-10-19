"""Tests for the advanced modules: conformal, risk, optimiser, analogs, bandit, copilot, readiness, API."""
import numpy as np
import pytest
from src import (conformal as cf, risk_analytics as ra, optimizer as op, analogs as an, bandit as bd, copilot as cp,
                 readiness as rd, scenario_engine as se, recommendation_engine as rec, data_ingestion as di, config)


# ---------------------------------------------------------------- conformal
def test_conformal_coverage_meets_target_on_synthetic_errors():
    rng = np.random.default_rng(1)
    lr = rng.normal(0, 0.5, 300)
    ex = cf.coverage_experiment(lr, alpha=0.2, n_rep=300, seed=3)
    assert ex["coverage"] >= 0.78      # finite-sample guarantee >= 0.8 up to Monte-Carlo error

def test_conformal_interval_ordering_and_small_n():
    lr = np.random.default_rng(0).normal(0, 0.4, 50)
    ci = cf.conformal_interval(1000.0, lr, 0.2)
    assert ci["finite"] and ci["low"] < 1000 * np.exp(np.median(lr)) < ci["high"]
    tiny = cf.conformal_interval(1000.0, lr[:4], 0.05)       # too few points for a 95 % interval -> unbounded, never fake
    assert not tiny["finite"]


# ---------------------------------------------------------------- risk
def test_tail_risk_identities():
    x = np.arange(-100, 101, dtype=float)
    t = ra.tail_risk(x, target=0.0, alpha=0.05)
    assert t["cvar"] <= t["var"] <= t["median"]
    assert abs(t["p_hurdle"] - (x >= 0).mean()) < 1e-12
    assert ra.risk_adjusted_value(x, lam=0.0) == pytest.approx(x.mean())
    assert ra.risk_adjusted_value(x, lam=1.0) == pytest.approx(t["cvar"])

def test_profit_simulation_matches_engine_hurdle_probability(ctx):
    inp = se.LaunchInputs()
    p = ra.simulate_profit(inp, ctx, ra.draw_noise(ctx, 3000, seed=11))
    ref = se.run_assessment(inp, ctx, n_sims=3000, seed=12)["p_hurdle"]
    assert abs((p >= inp.target_profit).mean() - ref) < 0.04

def test_sobol_indices_in_range_and_distribution_sensible(ctx):
    s = ra.sobol_indices(se.LaunchInputs(), ctx, n=16)
    assert set(s.driver) == set(ra.SOBOL_FACTORS)
    assert s.ST.between(0, 1.2).all() and s.S1.between(0, 1).all()
    assert s.ST.max() > 0.2          # something must matter


# ---------------------------------------------------------------- optimiser
def test_dp_equals_greedy_and_beats_naive_split():
    for total in (400_000, 1_000_000, 1_600_000):
        dp, gr = op.allocate_budget_dp(total), op.allocate_greedy(total)
        assert dp["value"] == pytest.approx(gr["value"], rel=1e-9)
        assert sum(dp["allocation"].values()) == pytest.approx(total)
        naive = {k: total / 4 for k in op.CHANNEL_PRIORS}
        assert dp["value"] >= sum(op.response(naive[k], *op.CHANNEL_PRIORS[k]) for k in naive) - 1e-9

def test_dp_matches_bruteforce_two_channels():
    pri = {"a": (1.0, 100.0), "b": (2.0, 300.0)}
    step, total = 50.0, 600.0
    best = max(op.response(i * step, *pri["a"]) + op.response(total - i * step, *pri["b"]) for i in range(int(total / step) + 1))
    assert op.allocate_budget_dp(total, pri, step)["value"] == pytest.approx(best)

def test_optimizer_never_worse_than_current_on_its_objective(ctx):
    inp = se.LaunchInputs()
    R = op.optimize_strategy(inp, ctx, lam=0.5, n_sims=300, maxiter=6, popsize=4)
    assert R["optimised"]["risk_adjusted"] >= R["current"]["risk_adjusted"] - 1e-6
    lo, hi = ctx["ratio_range"]
    assert lo - 1e-9 <= R["x"]["Shelf price (x plan)"] <= hi + 1e-9      # never leaves the price range the evidence supports


# ---------------------------------------------------------------- analogs
def test_analogs_prefer_same_category_and_weights_sum_to_one(ctx, panel):
    lib = an.build_library(panel, di.load_catalog())
    inp = se.LaunchInputs()
    nn = an.find_analogs(inp, lib, 5)
    n_same = int((lib["attrs"].category == inp.category).sum())
    assert (nn.category == inp.category).sum() == min(5, n_same)                      # every same-category launch is retrieved ...
    assert list(nn.same_category[: min(5, n_same)]) == [True] * min(5, n_same)       # ... and ranks before any other category
    assert nn.weight.sum() == pytest.approx(1.0) and nn.distance.is_monotonic_increasing
    f = an.analog_forecast(inp, lib, nn)
    assert f["total"] > 0 and len(f["weekly"]) == an.WEEKS

def test_ensemble_check_flags_big_gaps():
    a = dict(total=100.0)
    assert an.ensemble_check(105.0, a)["agreement"] == "Strong"
    assert an.ensemble_check(300.0, a)["agreement"] == "Weak"


# ---------------------------------------------------------------- bandit
def test_thompson_beats_uniform_on_regret():
    pri = np.array([30.0, 36.0, 42.0, 39.0])
    out = bd.simulate_policies(pri, 0.3, 0.5, rounds=12, batch=20, n_runs=600, seed=5)
    assert out["Thompson sampling"]["cum_regret"][-1] < out["Uniform"]["cum_regret"][-1]
    assert out["Thompson sampling"]["p_identify_best"] > 0.5
    assert bd.posterior_best_probability(pri).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- readiness + copilot
def test_readiness_scores_bounded(ctx):
    res = se.run_assessment(se.LaunchInputs(), ctx, n_sims=300)
    sc = rd.scores(res, ctx)
    assert set(sc) == set(rd.WEIGHTS) and all(0 <= v <= 100 for v in sc.values())
    assert sum(rd.WEIGHTS.values()) == pytest.approx(1.0) and 0 <= rd.composite(sc) <= 100

def test_copilot_grounding_guard(ctx):
    inp = se.LaunchInputs(); res = se.run_assessment(inp, ctx, n_sims=300)
    f = cp.build_facts(res, rec.decide(res), rec.generate(res, ctx, None, None))
    assert cp.answer(f, "why this decision?")["check"]["grounded"]
    assert cp.grounding_check(f"P(hurdle) is {f['p_hurdle_pct']:.1f}%", f)["grounded"]
    bad = cp.grounding_check("Expected profit is 987654321 and 77.7% chance", f)
    assert not bad["grounded"] and "987654321" in bad["ungrounded_numbers"]
    assert cp.answer(f, "what is the downside risk?")["source"].startswith("offline")


# ---------------------------------------------------------------- red team
def test_red_team_breakpoints_and_joint_search(ctx):
    from src import red_team as rt
    inp = se.LaunchInputs()
    noise = ra.draw_noise(ctx, 500, 7)
    bp = rt.breakpoints(inp, ctx, 0.5, 500, 7)
    assert bp.reached.any()
    for _, r in bp[bp.reached].iterrows():                         # bisection result: fails at breakpoint, holds just below it
        assert rt._p(inp, ctx, noise, {r.key: r.breakpoint}) <= 0.5 + 0.02
        assert rt._p(inp, ctx, noise, {r.key: 0.8 * r.breakpoint}) > 0.5 - 0.02
    R = rt.reverse_stress_test(inp, ctx, 0.5, n_sims=500, maxiter=6, seed=7)
    assert R["found"] and R["p_at_failure"] <= 0.52 and R["z"] <= bp.z.min() + 1e-9      # never worse than the best single-factor breakpoint
    assert sum(R["shares"].values()) == pytest.approx(1.0) and "missed its hurdle" in rt.pre_mortem(R, inp)

def test_red_team_distance_properties(ctx):
    from src import red_team as rt
    sig = rt.sigmas(ctx)
    assert rt.zdist({}, sig) == 0
    assert rt.zdist({"mfg_cost_pct": 12.0}, sig) == pytest.approx(2.0)                   # 12% / 6% sigma
    assert rt.zdist({"mfg_cost_pct": 6.0, "price_pct": 5.0}, sig) == pytest.approx(np.sqrt(2))

