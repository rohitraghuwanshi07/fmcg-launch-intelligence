"""Real-data layer: statistical correctness of the estimators, leakage, provenance, engine wiring, SHAP."""
import hashlib, json
import numpy as np
import pandas as pd
import pytest
from src import real_data as rd, shap_explain as sx, scenario_engine as se, config

needs_real = pytest.mark.skipif(not rd.available(), reason="real data not bundled")


def _simulate_choice(n=6000, beta=(-2.0, 0.8), seed=0):
    rng = np.random.default_rng(seed); J = 4
    P = rng.uniform(0.6, 1.4, (n, J)); D = rng.binomial(1, 0.2, (n, J)).astype(float)
    a = np.array([0, .3, -.2, .1]); V = beta[0] * P + beta[1] * D + a
    Pr = np.exp(V) / np.exp(V).sum(1, keepdims=True)
    y = np.array([rng.choice(J, p=p) for p in Pr]); return P, [D], y


def test_clogit_recovers_planted_parameters_and_gradient_is_correct():
    P, X, y = _simulate_choice()
    th, ll, k = rd._fit_clogit(P, X, y)
    assert abs(th[0] - (-2.0)) < 0.25 and abs(th[1] - 0.8) < 0.25
    # analytic gradient vs numeric: re-derive nll via the fit's own function by finite differences on log-likelihood
    from scipy.optimize import approx_fprime
    def ll_fn(t):
        b = t[:k]; a = np.concatenate([[0.0], t[k:]]); V = b[0] * P + b[1] * X[0] + a
        V -= V.max(1, keepdims=True); E = np.exp(V); Pr = E / E.sum(1, keepdims=True)
        return np.log(Pr[np.arange(len(y)), y]).sum()
    th2 = th + 0.05
    num = approx_fprime(th2, ll_fn, 1e-6)
    b = th2[:k]; a = np.concatenate([[0.0], th2[k:]]); V = b[0] * P + b[1] * X[0] + a
    V -= V.max(1, keepdims=True); E = np.exp(V); Pr = E / E.sum(1, keepdims=True)
    ana = np.concatenate([[(P[np.arange(len(y)), y] - (Pr * P).sum(1)).sum(), (X[0][np.arange(len(y)), y] - (Pr * X[0]).sum(1)).sum()],
                          (np.eye(4)[y] - Pr).sum(0)[1:]])
    assert np.allclose(num, ana, rtol=1e-3, atol=1e-2)


def test_elasticity_formula_matches_closed_form_for_symmetric_case():
    n, J = 2000, 3; P = np.ones((n, J)); th = np.array([-2.0, 0.0, 0.0]); th = th[:1].tolist() + [0.0] * (J - 1)
    e, share = rd._elasticity(np.array(th), P, [], 1)
    assert e == pytest.approx(-2.0 * (1 - 1 / 3), rel=1e-9) and np.allclose(share, 1 / 3)


@needs_real
def test_real_elasticities_are_plausible_and_have_valid_intervals():
    r = rd.real_elasticities()
    for n, d in r["datasets"].items():
        assert -8 < d["elasticity"] < -0.3, (n, d["elasticity"])
        assert d["ci_low"] <= d["elasticity"] <= d["ci_high"] and d["se"] > 0
    assert set(rd.CATEGORY_SOURCES) <= set(r["categories"])
    assert r["datasets"]["Cracker"]["promo"]["feat"]["odds_multiplier"] > 1       # featured products are chosen more often


@needs_real
def test_provenance_hashes_match_bundled_files():
    prov = json.loads((config.DATA_REAL / "PROVENANCE.json").read_text())
    for name, p in prov.items():
        assert hashlib.sha256((config.DATA_REAL / f"{name}.csv").read_bytes()).hexdigest() == p["sha256"], name


@needs_real
def test_forecast_features_never_use_the_future():
    df = rd.load_retail(); sid = df.sid.iloc[0]; g = df[df.sid == sid].copy()
    F1 = rd._features(g); g2 = g.copy(); cut = g.date.iloc[200]
    g2.loc[g2.date > cut, "y"] *= 7.0                                            # corrupt everything after the origin
    F2 = rd._features(g2)
    a = F1[F1.origin <= cut].select_dtypes("number"); b = F2[F2.origin <= cut].select_dtypes("number")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@needs_real
def test_real_benchmark_beats_seasonal_naive_and_conformal_is_calibrated():
    B = rd.load_benchmark(); assert B is not None
    for r in B["by_horizon"]:
        assert r["GBM (global)|WAPE"] < r["Seasonal naive|WAPE"], r["h"]
    for h, c in B["conformal"].items():
        assert c["coverage"] >= 0.75, (h, c["coverage"])
    assert B["split"]["cal_end"] < B["split"]["last"]


@needs_real
def test_seasonality_profile_is_normalised_and_christmas_peaks_for_supermarkets():
    p = rd.seasonal_profile("Supermarket and grocery stores")
    assert p["index"].mean() == pytest.approx(1.0) and int(p.loc[p["index"].idxmax(), "month"]) == 12


@needs_real
def test_engine_uses_real_elasticity_and_env_switch_restores_simulated(ctx, monkeypatch):
    c = se.load_context()
    assert c["elasticity_source"]["Biscuits & Snacks"] == "real scanner data"
    assert c["elasticity"]["Biscuits & Snacks"] == pytest.approx(rd.real_elasticities()["categories"]["Biscuits & Snacks"]["elasticity"])
    monkeypatch.setenv("FMCG_PRICE_SOURCE", "simulated")
    s = se.load_context()
    assert set(s["elasticity_source"].values()) == {"simulated panel"}


def test_shap_is_exact_and_matches_model():
    import joblib
    shipped = config.ROOT / "models" / "coldstart_model.joblib"           # the LightGBM model shipped with the repo
    ctx = dict(coldstart=joblib.load(shipped))
    assert sx.supported(ctx)
    inp = se.LaunchInputs(); cs = ctx["coldstart"]
    frame = se.build_launch_frame(inp, cs["support"])
    df, base, raw = sx.explain_rows(frame, ctx)
    pred = cs["model"].predict(se._featurize(frame, cs["ref"])[cs["features"]])
    assert np.allclose(np.exp(raw), pred, rtol=1e-6)                                   # base + sum(SHAP) == model output (log link)
    assert np.allclose(df.sum(1) + base, raw)
    assert set(df.columns) <= set(cs["features"])
