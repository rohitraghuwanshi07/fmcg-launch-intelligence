"""REAL-data layer (everything here is estimated from real, bundled datasets - see data/real/PROVENANCE.json).

1. Choice models on real US supermarket scanner panels (crackers, ketchup, tuna, yogurt, orange juice):
   conditional logit by maximum likelihood (analytic gradient), own-price elasticities with household-cluster bootstrap
   CIs, and promotion (display/feature) odds multipliers.
2. Real retail time series (Australian Bureau of Statistics, 152 state x industry series, 1982-2018):
   global direct multi-horizon gradient-boosting forecaster vs seasonal-naive / naive / moving average /
   Holt-Winters / SARIMA, conformal intervals, real seasonality profiles.
Caveat that is shown in the UI: scanner panels are brand-choice data from historical US supermarkets and the
retail series is Australian macro turnover - they inform price response and seasonality, they do not contain
new-product launches, so launch-curve (cold-start) accuracy still cannot be validated on real data here.
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import config

REAL = config.DATA_REAL
CACHE = REAL / "derived"
CATEGORY_SOURCES = {  # app category -> real scanner datasets that inform its price response
    "Biscuits & Snacks": ["Cracker"], "Dairy & Related": ["Yogurt"], "Packaged Grocery": ["Catsup", "Ketchup", "Tuna"],
    "Beverages & Juices": ["OJ"],
}
SEASON_INDUSTRY = {c: "Supermarket and grocery stores" for c in config.CATEGORIES} if hasattr(config, "CATEGORIES") else {}


def available() -> bool:
    return (REAL / "Cracker.csv").exists() and (REAL / "aus_retail.csv").exists()


# ----------------------------------------------------------------------------- 1. choice models
def _wide(name: str):
    """Return (alts, price[n,J], extras{name:[n,J]}, choice_idx[n], group[n])."""
    d = pd.read_csv(REAL / f"{name}.csv")
    if name == "OJ":
        alts = ["CH", "MM"]
        price = d[["SalePriceCH", "SalePriceMM"]].to_numpy(float)
        ex = {"special": d[["SpecialCH", "SpecialMM"]].to_numpy(float)}
        y = (d.Purchase == "MM").astype(int).to_numpy()
        return alts, price, ex, y, d.StoreID.to_numpy()
    if name in ("Ketchup", "Tuna"):
        pre = name + "."
        pc = [c for c in d.columns if c.startswith("price.")]; alts = [c.split(".", 1)[1] for c in pc]
        y = d[pre + "choice"].map({a: i for i, a in enumerate(alts)}).to_numpy()
        return alts, d[pc].to_numpy(float), {}, y, d[pre + "hid"].to_numpy()
    pc = [c for c in d.columns if c.startswith("price.")]; alts = [c.split(".", 1)[1] for c in pc]
    ex = {}
    for kind in ("disp", "feat"):
        cols = [f"{kind}.{a}" for a in alts]
        if all(c in d.columns for c in cols): ex[kind] = d[cols].to_numpy(float)
    y = d["choice"].map({a: i for i, a in enumerate(alts)}).to_numpy()
    return alts, d[pc].to_numpy(float), ex, y, d["id"].to_numpy()


def _fit_clogit(P, X, y, start=None):
    """Conditional logit: U_ij = a_j + b_p * price_ij + sum_m b_m x_ijm. Returns params [b_p, b_x..., a_1..a_{J-1}]."""
    n, J = P.shape; M = [P] + list(X); k = len(M)
    def unpack(th): return th[:k], np.concatenate([[0.0], th[k:]])
    def nll(th):
        b, a = unpack(th)
        V = sum(bi * m for bi, m in zip(b, M)) + a[None, :]
        V = V - V.max(1, keepdims=True); E = np.exp(V); Pr = E / E.sum(1, keepdims=True)
        ll = np.log(Pr[np.arange(n), y] + 1e-12).sum()
        g_b = np.array([sum(m[i, y[i]] for i in range(0)) if False else (m[np.arange(n), y] - (Pr * m).sum(1)).sum() for m in M])
        g_a = (np.eye(J)[y] - Pr).sum(0)[1:]
        return -ll, -np.concatenate([g_b, g_a])
    th0 = start if start is not None else np.zeros(k + J - 1)
    r = minimize(nll, th0, jac=True, method="L-BFGS-B")
    return r.x, -r.fun, k


def _elasticity(th, P, X, k):
    n, J = P.shape; b = th[:k]; a = np.concatenate([[0.0], th[k:]])
    V = b[0] * P + sum(bi * m for bi, m in zip(b[1:], X)) + a[None, :]
    V = V - V.max(1, keepdims=True); E = np.exp(V); Pr = E / E.sum(1, keepdims=True)
    eps_ij = b[0] * P * (1 - Pr)                    # own-price elasticity per observation and alternative
    share = Pr.mean(0)
    return float((eps_ij.mean(0) * share).sum() / share.sum()), share


def fit_dataset(name: str, n_boot: int = 40, seed: int = config.SEED) -> dict:
    alts, P, ex, y, grp = _wide(name)
    keep = np.isfinite(P).all(1) & (y >= 0)
    P, y, grp = P[keep], y[keep].astype(int), grp[keep]; ex = {k: v[keep] for k, v in ex.items()}
    scale = float(P.mean()); Ps = P / scale                     # estimate in scale-free price units
    X = list(ex.values())
    th, ll, k = _fit_clogit(Ps, X, y)
    eps, share = _elasticity(th, Ps, X, k)
    rng = np.random.default_rng(seed)
    ug = np.unique(grp); idx_by = {g: np.where(grp == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        try:
            tb, _, _ = _fit_clogit(Ps[ii], [x[ii] for x in X], y[ii], start=th)
            boots.append(_elasticity(tb, Ps[ii], [x[ii] for x in X], k)[0])
        except Exception:
            continue
    boots = np.array(boots)
    se = float(boots.std()) if len(boots) > 5 else float("nan")
    promo = {}
    for i, nm in enumerate(ex, start=1):
        promo[nm] = dict(odds_multiplier=float(np.exp(th[i] )), coef=float(th[i]),
                         exposure_share=float(ex[nm].mean()))
    return dict(dataset=name, n_obs=int(len(y)), n_alternatives=len(alts), alternatives=alts, elasticity=eps, se=se,
                ci_low=float(np.percentile(boots, 2.5)) if len(boots) > 5 else float("nan"),
                ci_high=float(np.percentile(boots, 97.5)) if len(boots) > 5 else float("nan"),
                price_coef_scaled=float(th[0]), loglik=float(ll), shares=dict(zip(alts, map(float, share))), promo=promo,
                n_clusters=int(len(ug)), n_boot=int(len(boots)))


def real_elasticities(refresh: bool = False) -> dict:
    """Own-price elasticity per real dataset and per app category (precision-weighted where several datasets inform it)."""
    f = CACHE / "elasticities.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    ds = {n: fit_dataset(n) for n in ("Cracker", "Catsup", "Ketchup", "Tuna", "Yogurt", "OJ")}
    cats = {}
    for cat, names in CATEGORY_SOURCES.items():
        e = np.array([ds[n]["elasticity"] for n in names]); se = np.array([max(ds[n]["se"], 1e-3) if np.isfinite(ds[n]["se"]) else 1.0 for n in names])
        w = 1 / se ** 2
        cats[cat] = dict(elasticity=float((e * w).sum() / w.sum()), se=float(1 / np.sqrt(w.sum())), sources=names)
    cat_e = np.array([v["elasticity"] for v in cats.values()])     # equal weight per category: precision-weighting would let the largest panel (tuna) dominate
    cats["pooled"] = dict(elasticity=float(cat_e.mean()), se=float(cat_e.std(ddof=1) / np.sqrt(len(cat_e))), sources=list(ds), note="unweighted mean of category estimates")
    out = dict(datasets=ds, categories=cats, note="Brand-level own-price elasticity from household choice data (includes switching from rival brands).")
    CACHE.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out, indent=2))
    return out


def elasticity_overrides() -> tuple[dict, dict]:
    """(elasticity by category, source label by category) for categories with real evidence; pooled real value is a fallback label."""
    if not available(): return {}, {}
    r = real_elasticities()["categories"]
    el = {c: v["elasticity"] for c, v in r.items()}
    return el, {c: "real scanner data" for c in el}


# ----------------------------------------------------------------------------- 2. real time series
def load_retail() -> pd.DataFrame:
    d = pd.read_csv(REAL / "aus_retail.csv")
    d["date"] = pd.to_datetime(d["Month"], format="%Y %b")
    d = d.rename(columns={"State": "state", "Industry": "industry", "Turnover": "y", "Series ID": "sid"})
    return d[["sid", "state", "industry", "date", "y"]].sort_values(["sid", "date"]).reset_index(drop=True)


H_LIST = tuple(range(1, 13))
LAGS = (0, 1, 2, 3, 5, 11, 12, 23)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, origin). All features use data up to and including the origin month only."""
    out = []
    for sid, g in df.groupby("sid", sort=False):
        g = g.sort_values("date").reset_index(drop=True); ly = np.log(g.y.clip(lower=1e-3))
        f = pd.DataFrame({"sid": sid, "origin": g.date, "ly": ly})
        for L in LAGS: f[f"lag{L}"] = ly.shift(L)
        f["level12"] = ly.rolling(12).mean(); f["level3"] = ly.rolling(3).mean(); f["vol12"] = ly.diff().rolling(12).std()
        f["yoy"] = ly - ly.shift(12); f["mom"] = ly - ly.shift(1)
        f["industry"] = g.industry.iloc[0]; f["state"] = g.state.iloc[0]
        out.append(f)
    F = pd.concat(out, ignore_index=True)
    F["ind_code"] = F.industry.astype("category").cat.codes; F["state_code"] = F.state.astype("category").cat.codes
    return F


def _with_target(F: pd.DataFrame, h: int) -> pd.DataFrame:
    G = F.copy()
    G["target_date"] = G.origin + pd.DateOffset(months=h)
    t = F[["sid", "origin", "ly"]].rename(columns={"origin": "target_date", "ly": "ly_target"})
    G = G.merge(t, on=["sid", "target_date"], how="left")
    tm = G.target_date.dt.month
    G["m_sin"] = np.sin(2 * np.pi * tm / 12); G["m_cos"] = np.cos(2 * np.pi * tm / 12); G["tmonth"] = tm
    G["y_resid"] = G.ly_target - G.level12                      # scale-free target
    return G


FEATS = [f"lag{L}" for L in LAGS] + ["level12", "level3", "vol12", "yoy", "mom", "ind_code", "state_code", "m_sin", "m_cos", "tmonth"]


def _rel(F, cols):
    X = F[cols].copy()
    for L in LAGS: X[f"lag{L}"] = F[f"lag{L}"] - F["level12"]
    X["level3"] = F["level3"] - F["level12"]
    return X


def run_forecast_benchmark(train_end="2012-12-01", cal_end="2015-06-01", seed=config.SEED, sarima_series: int = 12) -> dict:
    import lightgbm as lgb
    df = load_retail(); F = _features(df)
    te, ce = pd.Timestamp(train_end), pd.Timestamp(cal_end)
    last = df.date.max()
    res = {"by_horizon": [], "baselines": {}, "conformal": {}, "split": dict(train_end=str(te.date()), cal_end=str(ce.date()), last=str(last.date()))}
    models = {}; preds_all = []
    for h in H_LIST:
        G = _with_target(F, h).dropna(subset=FEATS[:-3] + ["ly_target", "y_resid"])
        G = G[G.target_date <= last]
        tr = G[G.target_date <= te]; ca = G[(G.target_date > te) & (G.target_date <= ce)]; ts = G[G.target_date > ce]
        Xcols = [c for c in FEATS]
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, objective="l1", random_state=seed, verbose=-1)
        mdl.fit(_rel(tr, Xcols), tr.y_resid, categorical_feature=["ind_code", "state_code"])
        models[h] = mdl
        for name, part in (("cal", ca), ("test", ts)):
            part = part.copy(); part["pred_ly"] = part.level12 + mdl.predict(_rel(part, Xcols)); part["split"] = name; part["h"] = h
            preds_all.append(part)
    P = pd.concat(preds_all, ignore_index=True)
    # baselines at the same (series, origin, h)
    ly_lookup = F.set_index(["sid", "origin"]).ly
    P["naive_ly"] = P.ly
    P["snaive_ly"] = [ly_lookup.get((s, o + pd.DateOffset(months=h - 12)), np.nan) if h <= 12 else np.nan for s, o, h in zip(P.sid, P.origin, P.h)]
    P["ma_ly"] = P.level12
    P = P.dropna(subset=["pred_ly", "ly_target", "snaive_ly"])
    # MASE scale: in-sample mean abs seasonal difference (training period) per series on original scale
    tr_df = df[df.date <= te]
    scale = tr_df.groupby("sid").y.apply(lambda s: np.mean(np.abs(s.values[12:] - s.values[:-12]))).rename("scale")
    P = P.merge(scale, on="sid")
    P["actual"] = np.exp(P.ly_target)
    for m in ("pred", "naive", "snaive", "ma"):
        P[f"f_{m}"] = np.exp(P[f"{m}_ly"])
    # conformal on calibration residuals per horizon (log scale, order-statistic interval)
    from .conformal import _order_stat_bounds
    cov_rows = []
    for h in H_LIST:
        cal = P[(P.split == "cal") & (P.h == h)]; tst = P[(P.split == "test") & (P.h == h)]
        r = (cal.ly_target - cal.pred_ly).to_numpy(); lo, hi = _order_stat_bounds(r, 0.2)
        inside = ((tst.ly_target - tst.pred_ly >= lo) & (tst.ly_target - tst.pred_ly <= hi)).mean()
        cov_rows.append(dict(h=h, lo=float(lo), hi=float(hi), coverage=float(inside), n_cal=int(len(cal)), n_test=int(len(tst))))
        res["conformal"][str(h)] = cov_rows[-1]
    T = P[P.split == "test"]
    def metrics(g, col):
        a, f = g.actual.values, g[col].values
        return dict(WAPE=float(np.abs(a - f).sum() / a.sum()), MASE=float(np.mean(np.abs(a - f) / g.scale.values)),
                    sMAPE=float(np.mean(2 * np.abs(a - f) / (np.abs(a) + np.abs(f))) ))
    for h in H_LIST:
        g = T[T.h == h]
        row = {"h": h, "n": int(len(g))}
        for m, lab in (("pred", "GBM (global)"), ("snaive", "Seasonal naive"), ("naive", "Naive"), ("ma", "12-m average")):
            for k, v in metrics(g, f"f_{m}").items(): row[f"{lab}|{k}"] = v
        res["by_horizon"].append(row)
    # per-industry test WAPE for h=3
    g3 = T[T.h == 3]
    res["by_industry_h3"] = (g3.groupby("industry").apply(lambda g: pd.Series(dict(
        gbm=np.abs(g.actual - g.f_pred).sum() / g.actual.sum(), snaive=np.abs(g.actual - g.f_snaive).sum() / g.actual.sum())), include_groups=False)
        .reset_index().to_dict("records"))
    imp = pd.Series(models[3].feature_importances_, index=FEATS).sort_values(ascending=False)
    res["importance_h3"] = imp.head(12).to_dict()
    res["n_series"] = int(df.sid.nunique()); res["n_test_origins_h3"] = int(g3.origin.nunique())
    # statistical benchmarks (Holt-Winters, SARIMA) on a sample of series, fixed origins, h = 1..12
    res["stat_benchmarks"] = _stat_benchmarks(df, F, models, sarima_series, last)
    return res, models, P


def _stat_benchmarks(df, F, models, n_series, last):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    rng = np.random.default_rng(config.SEED)
    sids = rng.choice(df.sid.unique(), n_series, replace=False)
    origins = [last - pd.DateOffset(months=m) for m in (36, 24, 12)]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sid in sids:
            g = df[df.sid == sid].set_index("date").y
            for org in origins:
                hist = g[g.index <= org]; fut = g[(g.index > org)].iloc[:12]
                if len(fut) < 12 or len(hist) < 60: continue
                lh = np.log(hist.clip(lower=1e-3)); out = {}
                try: out["Holt-Winters"] = np.exp(ExponentialSmoothing(lh.values, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit().forecast(12))
                except Exception: pass
                try: out["SARIMA"] = np.exp(SARIMAX(lh.values, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=60).forecast(12))
                except Exception: pass
                out["Seasonal naive"] = hist.values[-12:]
                pts = []
                for h in range(1, 13):                  # one direct model per horizon, no interpolation
                    G = _with_target(F[(F.sid == sid)], h); G = G[G.origin == org]
                    pts.append(float(G.level12.iloc[0] + models[h].predict(_rel(G, FEATS))[0]) if len(G) else np.nan)
                if np.isfinite(pts).all():
                    out["GBM (global)"] = np.exp(pts)
                for k, v in out.items():
                    rows.append(dict(sid=sid, origin=str(org.date()), model=k, wape=float(np.abs(fut.values - v).sum() / fut.values.sum())))
    t = pd.DataFrame(rows)
    if t.empty: return []
    both = t.groupby(["sid", "origin"]).model.nunique(); ok = both[both == t.model.nunique()].index
    t = t.set_index(["sid", "origin"]).loc[ok].reset_index()
    return t.groupby("model").wape.agg(["mean", "median", "count"]).reset_index().to_dict("records")


def seasonal_profile(industry: str | None = None) -> pd.DataFrame:
    """Real seasonal index per month-of-year: turnover / centered 12-month mean, averaged over 1995-2017 and states."""
    df = load_retail()
    df = df[df.industry == industry] if industry else df
    out = []
    for sid, g in df.groupby("sid"):
        s = g.set_index("date").y; c = s.rolling(13, center=True).mean()
        r = (s / c).dropna(); r = r[(r.index.year >= 1995) & (r.index.year <= 2017)]
        out.append(r.groupby(r.index.month).mean().rename(sid))
    prof = pd.concat(out, axis=1).mean(axis=1); prof = prof / prof.mean()
    return prof.rename("index").rename_axis("month").reset_index()


def seasonal_context(category: str, launch_date: str, weeks: int = 26) -> dict:
    """Real seasonality over the launch window for the app category's retail channel."""
    ind = "Pharmaceutical, cosmetic and toiletry goods retailing" if category == "Personal Care" else "Supermarket and grocery stores"
    prof = seasonal_profile(ind).set_index("month")["index"]
    start = pd.Timestamp(launch_date); months = [(start + pd.Timedelta(weeks=w)).month for w in range(weeks)]
    window = float(np.mean([prof[m] for m in months]))
    return dict(industry=ind, window_index=window, start_index=float(prof[start.month]), peak_month=int(prof.idxmax()), trough_month=int(prof.idxmin()),
                profile=prof.to_dict())


def _final_forecast(F, df, last, conformal) -> pd.DataFrame:
    """Refit direct models on ALL data and forecast the 12 months after the last observation, with conformal bands."""
    import lightgbm as lgb
    rows = []
    for h in H_LIST:
        G = _with_target(F, h)
        tr = G.dropna(subset=FEATS[:-3] + ["ly_target", "y_resid"]); tr = tr[tr.target_date <= last]
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, objective="l1", random_state=config.SEED, verbose=-1)
        mdl.fit(_rel(tr, FEATS), tr.y_resid, categorical_feature=["ind_code", "state_code"])
        cur = G[G.origin == last].dropna(subset=FEATS[:-3])
        pred = cur.level12 + mdl.predict(_rel(cur, FEATS)); lo, hi = conformal[str(h)]["lo"], conformal[str(h)]["hi"]
        rows.append(pd.DataFrame(dict(sid=cur.sid.values, h=h, target_date=cur.target_date.values, forecast=np.exp(pred.values),
                                      lo=np.exp(pred.values + lo), hi=np.exp(pred.values + hi))))
    return pd.concat(rows, ignore_index=True)


def build_real_artifacts() -> dict:
    """Run everything real and cache results in data/real/derived (used by the Real Data Lab page)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    el = real_elasticities(refresh=True)
    res, models, P = run_forecast_benchmark()
    df = load_retail(); F = _features(df); last = df.date.max()
    fut = _final_forecast(F, df, last, res["conformal"])
    P[P.split == "test"][["sid", "origin", "h", "actual", "f_pred", "f_snaive"]].to_csv(CACHE / "test_predictions.csv.gz", index=False)
    fut.to_csv(CACHE / "future_forecast.csv.gz", index=False)
    (CACHE / "forecast_benchmark.json").write_text(json.dumps(res, indent=2, default=float))
    return dict(elasticities=el, benchmark=res)


def load_benchmark() -> dict | None:
    f = CACHE / "forecast_benchmark.json"
    return json.loads(f.read_text()) if f.exists() else None
