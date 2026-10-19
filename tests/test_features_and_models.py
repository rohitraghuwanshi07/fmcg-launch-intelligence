import numpy as np
import pandas as pd
import pytest
from src import feature_engineering as fe, forecasting as fc, pricing_analytics as pa, demo_data, config


def test_no_temporal_leakage_in_lag_features(panel):
    """Changing units at/after a cutoff must not change features of rows whose forecast origin precedes the cutoff."""
    h = config.HORIZON_WEEKS
    ref = fe.fit_category_reference(panel)
    cutoff = panel.date.sort_values().unique()[70]
    base = fe.build_lag_features(panel, ref, h)
    pert = panel.copy(); pert.loc[pert.date >= cutoff, "units"] = pert.loc[pert.date >= cutoff, "units"] * 7 + 13
    alt = fe.build_lag_features(pert, ref, h)
    safe = base.date <= cutoff + pd.Timedelta(weeks=h - 1)
    cols = ["lag_h", "lag_h1", "lag_h4", "lag_52", "roll_mean_4", "roll_mean_8", "roll_std_8", "ewm_03", "price_ratio_recent"]
    a, b = base.loc[safe, cols].to_numpy(float), alt.loc[safe, cols].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)
    assert not np.allclose(base.loc[~safe, cols].to_numpy(float), alt.loc[~safe, cols].to_numpy(float), equal_nan=True)  # sanity: later rows do change

def test_lag_features_reject_gappy_series(panel):
    gappy = panel.drop(panel.index[panel.groupby(["product_id", "store_id"]).cumcount() == 5])
    with pytest.raises(ValueError):
        fe.build_lag_features(gappy, fe.fit_category_reference(panel))

def test_coldstart_frame_only_observed_launches(panel):
    cs = fe.build_coldstart_frame(panel, fe.fit_category_reference(panel))
    assert cs.launch_observed.all() and cs.age_weeks.max() < config.COLD_START_WEEKS
    assert fe.build_coldstart_frame(panel.drop(columns=["launch_observed"]), {}).empty

def test_lag_forecaster_beats_baselines_chronologically(trained):
    b = fc.load_bundle("lag_forecaster")
    t = b["metrics"]["test"]; sel = [k for k in t if "[selected]" in k][0]
    best_baseline = min(v["WAPE"] for k, v in t.items() if k != sel)
    assert t[sel]["WAPE"] < best_baseline
    assert b["train_end"] < b["val_end"]
    assert 0.65 < b["interval_coverage_test"] < 0.92        # nominal 80%

def test_coldstart_bundle_and_grouped_validation(trained):
    b = fc.load_bundle("coldstart_model")
    assert b["metrics"]["cv_dev"]["Baseline"] and b["n_ratio_products"] >= 20
    assert np.isfinite(b["log_ratio"]).all() and 0 < np.std(b["log_ratio"]) < 2
    assert len(b["test_products"]) >= 4

def test_grouped_cv_never_shares_products():
    from sklearn.model_selection import GroupKFold
    g = np.repeat(np.arange(20), 10)
    for tr, te in GroupKFold(5).split(g, groups=g):
        assert not set(g[tr]) & set(g[te])

def test_prediction_pipeline_is_column_order_invariant(ctx):
    from src import scenario_engine as se
    cs = ctx["coldstart"]; inp = se.LaunchInputs()
    f = se._featurize(se.build_launch_frame(inp, cs["support"]), cs["ref"])
    X = f[cs["features"]]
    p1 = cs["model"].predict(X); p2 = cs["model"].predict(X[X.columns[::-1]])
    assert np.allclose(p1, p2)

def test_elasticity_recovers_planted_values(panel):
    e = pa.estimate_elasticity(panel).set_index("scope")
    for cat, true in demo_data.PLANTED_ELASTICITY.items():
        if np.isfinite(e.loc[cat, "elasticity"]):
            assert abs(e.loc[cat, "elasticity"] - true) < 0.45, (cat, e.loc[cat, "elasticity"], true)
    assert -2.3 < e.loc["All categories (pooled)", "elasticity"] < -1.1

def test_forecast_metrics_math():
    m = fc.forecast_metrics([10, 20, 0], [12, 18, 0], scale=2)
    assert m["MAE"] == pytest.approx(4 / 3) and m["WAPE"] == pytest.approx(4 / 30) and m["MASE"] == pytest.approx(2 / 3)
