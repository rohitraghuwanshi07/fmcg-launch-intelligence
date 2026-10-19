import math
import numpy as np
import pytest
from dataclasses import replace
from src import scenario_engine as se, recommendation_engine as rec, marketing_analytics as ma


def test_unit_economics_hand_calculation():
    i = se.LaunchInputs(price=112, tax_rate=0.12, trade_margin=0.20, mfg_cost=30, packaging_cost=5, dist_cost=5, wastage_pct=0.0,
                        intro_discount=0.10, discount_funded_share=1.0)
    reg, pro = se.unit_economics(i, False), se.unit_economics(i, True)
    assert reg["price_ex_tax"] == pytest.approx(100) and reg["trade_margin_amt"] == pytest.approx(20)
    assert reg["net_revenue"] == pytest.approx(80) and reg["contribution"] == pytest.approx(40)
    assert pro["discount_cost"] == pytest.approx(10) and pro["net_revenue"] == pytest.approx(70) and pro["contribution"] == pytest.approx(30)
    assert pro["trade_margin_amt"] == reg["trade_margin_amt"]                    # discount not double counted in trade margin
    assert pro["consumer_price"] == pytest.approx(112 * 0.9)
    half = se.unit_economics(replace(i, discount_funded_share=0.5), True)
    assert half["net_revenue"] == pytest.approx(75)

def test_break_even_volume():
    assert se.break_even_volume(400_000, 40) == 10_000
    assert math.isinf(se.break_even_volume(1000, 0)) and math.isinf(se.break_even_volume(1000, -3))

def test_assessment_outputs_are_coherent(ctx):
    r = se.run_assessment(se.LaunchInputs(), ctx)
    assert 0 <= r["p_hurdle"] <= 1
    u = r["horizons"].units_mean.to_numpy(); assert np.all(np.diff(u) > 0)
    h = r["horizons"]; assert (h.units_p10 <= h.units_p50 + 1e-6).all() and (h.units_p50 <= h.units_p90 + 1e-6).all()
    s = r["scenarios"].set_index("scenario")
    assert s.loc["Conservative", "units_6m"] <= s.loc["Base", "units_6m"] <= s.loc["Optimistic", "units_6m"] + 1e-6
    assert r["inventory"]["p_stockout"] >= 0 and r["reliability"]["level"] in ("High", "Moderate", "Low")

def test_fixed_costs_reduce_hurdle_probability(ctx):
    base = se.LaunchInputs()
    lo = se.run_assessment(base, ctx, n_sims=800)["p_hurdle"]
    hi = se.run_assessment(replace(base, other_fixed_costs=base.other_fixed_costs + 6_000_000, total_launch_budget=9e6), ctx, n_sims=800)["p_hurdle"]
    assert hi < lo

def test_price_increase_lowers_units_through_elasticity(ctx):
    base = se.LaunchInputs()
    u0 = se.run_assessment(base, ctx, mc=False)["horizons"].set_index("horizon").loc["6 months", "units_mean"]
    u1 = se.run_assessment(se.apply_whatif(base, price_pct=10), ctx, mc=False)["horizons"].set_index("horizon").loc["6 months", "units_mean"]
    u2 = se.run_assessment(se.apply_whatif(base, price_pct=-10), ctx, mc=False)["horizons"].set_index("horizon").loc["6 months", "units_mean"]
    assert u1 < u0 < u2

def test_supply_constraint_caps_sales(ctx):
    i = se.LaunchInputs(initial_inventory=1000, weekly_capacity=0, lead_time_weeks=1)
    r = se.run_assessment(i, ctx, mc=True, n_sims=500)
    assert r["horizons"].units_mean.iloc[-1] <= 1000 + 1e-6 and r["inventory"]["p_stockout"] > 0.9

def test_invalid_and_unsupported_inputs_raise(ctx):
    with pytest.raises(ValueError): se.run_assessment(se.LaunchInputs(price=-1), ctx)
    with pytest.raises(ValueError): se.run_assessment(se.LaunchInputs(category="Pet Food"), ctx)
    with pytest.raises(ValueError): se.run_assessment(se.LaunchInputs(regions=["Atlantis"]), ctx)

def test_marketing_response_is_monotone_and_saturating():
    v = [ma.uplift_from_spend(s, 0.2, 1000) for s in (0, 500, 1000, 5000, 50_000)]
    assert v[0] == 0 and all(np.diff(v) > 0) and v[-1] < 0.2 and v[2] == pytest.approx(0.1)

def test_what_if_table_has_all_standard_scenarios(ctx):
    base = se.LaunchInputs()
    t = se.what_if_table(base, ctx, n_sims=300)
    assert len(t) == len(se.STANDARD_WHATIFS) + 1 and t.p_hurdle.between(0, 1).all()

def test_recommendations_respond_to_inputs(ctx):
    good = se.run_assessment(se.LaunchInputs(price=45), ctx)
    thin = se.run_assessment(se.LaunchInputs(price=20, mfg_cost=11, target_margin=0.35), ctx)
    areas_good = {r.area + r.action for r in rec.generate(good, ctx)}
    recs_thin = rec.generate(thin, ctx)
    assert any("Raise shelf price" in r.action for r in recs_thin) and not any("Raise shelf price" in a for a in areas_good)
    for r in recs_thin:
        assert r.reason and r.evidence and r.tradeoff and r.source in ("model-derived", "rule-based", "manual-assumption") and r.confidence in ("High", "Medium", "Low")

def test_decision_reconsider_when_contribution_negative(ctx):
    r = se.run_assessment(se.LaunchInputs(mfg_cost=30), ctx)
    assert rec.decide(r)["decision"] == "Reconsider the launch"

def test_demo_reliability_is_capped(ctx):
    r = se.run_assessment(se.LaunchInputs(), ctx)
    assert r["reliability"]["level"] == "Low" and any("DEMO" in f for _, f in r["reliability"]["flags"])
