"""Model A: demand forecasting.

A1  Lag-based direct 4-week-ahead forecaster for EXISTING series (chronological train/val/test).
A2  Cold-start launch-demand model for NEW products (no sales history; validated on UNSEEN products via
    grouped cross-validation + a held-out product test set).
"""
from __future__ import annotations
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from . import config
from .feature_engineering import (CS_CAT, CS_NUM, LAG_CAT, LAG_NUM, build_lag_features,
                                  build_coldstart_frame, fit_category_reference)

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False


# ------------------------------------------------------------------ metrics
def forecast_metrics(y, p, scale=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    mae = float(np.mean(np.abs(err)))
    out = dict(MAE=mae, RMSE=float(np.sqrt(np.mean(err ** 2))), WAPE=float(np.abs(err).sum() / max(y.sum(), 1e-9)),
               bias_pct=float(err.sum() / max(y.sum(), 1e-9) * 100), n=int(len(y)))
    if scale and scale > 0:
        out["MASE"] = mae / scale
    return out


# ------------------------------------------------------------------ model factories
def _prep(cat, num, scale=False):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("num", StandardScaler() if scale else "passthrough", num)])

def lag_candidates(seed=config.SEED):
    m = {
        "Random Forest": Pipeline([("p", _prep(LAG_CAT, LAG_NUM)), ("i", SimpleImputer(strategy="median")),
                                   ("m", RandomForestRegressor(150, min_samples_leaf=10, n_jobs=-1, random_state=seed))]),
        "HistGradientBoosting (Poisson)": Pipeline([("p", _prep(LAG_CAT, LAG_NUM)),
                                   ("m", HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.06,
                                                                      max_leaf_nodes=20, min_samples_leaf=40, random_state=seed))]),
    }
    if HAS_LGBM:
        m["LightGBM (Poisson)"] = Pipeline([("p", _prep(LAG_CAT, LAG_NUM)),
                                  ("m", LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05, num_leaves=15,
                                                      min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                      random_state=seed, verbose=-1, n_jobs=-1))])
    return m

def cs_candidates(seed=config.SEED):
    m = {
        "Poisson GLM": Pipeline([("p", _prep(CS_CAT, CS_NUM, scale=True)), ("m", PoissonRegressor(alpha=1e-3, max_iter=1000))]),
        "Random Forest": Pipeline([("p", _prep(CS_CAT, CS_NUM)),
                                   ("m", RandomForestRegressor(150, min_samples_leaf=20, n_jobs=-1, random_state=seed))]),
        "HistGradientBoosting (Poisson)": Pipeline([("p", _prep(CS_CAT, CS_NUM)),
                                   ("m", HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.06,
                                                                      max_leaf_nodes=20, min_samples_leaf=40, random_state=seed))]),
    }
    if HAS_LGBM:
        m["LightGBM (Poisson)"] = Pipeline([("p", _prep(CS_CAT, CS_NUM)),
                                  ("m", LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05, num_leaves=15,
                                                      min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                                      random_state=seed, verbose=-1, n_jobs=-1))])
    return m


class GroupMeanBaseline:
    """Baseline for cold start: mean units by (category, store_type, age bucket)."""
    BINS = [-1, 3, 12, 26, 1000]
    def _key(self, X):
        b = pd.cut(X["age_weeks"], self.BINS, labels=False)
        return pd.DataFrame({"category": X["category"].values, "store_type": X["store_type"].values, "b": b.values}, index=X.index)
    def fit(self, X, y):
        k = self._key(X)
        self.mean_ = float(np.mean(y))
        self.tab_ = pd.Series(np.asarray(y), index=X.index).groupby([k.category, k.store_type, k.b]).mean()
        return self
    def predict(self, X):
        k = self._key(X)
        idx = pd.MultiIndex.from_frame(k[["category", "store_type", "b"]])
        return self.tab_.reindex(idx).fillna(self.mean_).to_numpy()


# ------------------------------------------------------------------ A1: lag-based forecaster
def chrono_split_dates(weeks):
    weeks = sorted(pd.to_datetime(weeks).unique())
    n = len(weeks)
    return weeks[int(0.70 * n) - 1], weeks[int(0.85 * n) - 1]

def train_lag_forecaster(panel: pd.DataFrame, seed=config.SEED, save=True) -> dict:
    h = config.HORIZON_WEEKS
    train_end, val_end = chrono_split_dates(panel["date"].unique())
    ref = fit_category_reference(panel[panel.date <= train_end])
    F = build_lag_features(panel, ref, h)
    F = F[F["lag_h"].notna()].copy()
    tr, va, te = F[F.date <= train_end], F[(F.date > train_end) & (F.date <= val_end)], F[F.date > val_end]
    if min(len(tr), len(va), len(te)) < 500:
        raise ValueError("Insufficient observations for a chronological train/val/test split.")
    scale = float(np.mean(np.abs(tr["units"] - tr["lag_h"])))
    feats = LAG_CAT + LAG_NUM
    baselines = {"Seasonal naive (lag 52)": "lag_52", f"Naive (lag {h})": "lag_h",
                 "Moving average (4w)": "roll_mean_4", "Exp. smoothing (alpha=0.3)": "ewm_03"}
    def base_pred(d, col):
        return d[col].fillna(d["roll_mean_4"]).fillna(d["lag_h"]).to_numpy()
    res = {"validation": {}, "test": {}}
    for name, col in baselines.items():
        res["validation"][name] = forecast_metrics(va.units, base_pred(va, col), scale)
    cands, vpred = lag_candidates(seed), {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mdl in cands.items():
            mdl.fit(tr[feats], tr["units"])
            vpred[name] = np.clip(mdl.predict(va[feats]), 0, None)
            res["validation"][name] = forecast_metrics(va.units, vpred[name], scale)
    best = min(cands, key=lambda n: res["validation"][n]["WAPE"])
    lr = np.log1p(va.units.to_numpy()) - np.log1p(vpred[best])
    q_lo, q_hi = float(np.quantile(lr, 0.10)), float(np.quantile(lr, 0.90))
    final = lag_candidates(seed)[best]
    trva = pd.concat([tr, va])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(trva[feats], trva["units"])
    pte = np.clip(final.predict(te[feats]), 0, None)
    for name, col in baselines.items():
        res["test"][name] = forecast_metrics(te.units, base_pred(te, col), scale)
    res["test"][best + " [selected]"] = forecast_metrics(te.units, pte, scale)
    lo, hi = np.expm1(np.log1p(pte) + q_lo), np.expm1(np.log1p(pte) + q_hi)
    coverage = float(np.mean((te.units.to_numpy() >= lo) & (te.units.to_numpy() <= hi)))
    seg = {c: forecast_metrics(g.units, pte[te.index.get_indexer(g.index)], scale)["WAPE"]
           for c, g in te.groupby("category") if len(g) >= 100}
    out = te[["date", "product_id", "store_id", "category", "region", "store_type", "units", "price"]].copy()
    out["pred"], out["pred_lo80"], out["pred_hi80"] = pte, lo, hi
    out["seasonal_naive"] = base_pred(te, "lag_52")
    out["moving_avg"] = base_pred(te, "roll_mean_4")
    bundle = dict(kind="lag_forecaster", model=final, model_name=best, features=feats, ref=ref, horizon=h,
                  interval_logq=(q_lo, q_hi), train_end=str(train_end.date()), val_end=str(val_end.date()),
                  metrics=res, interval_coverage_test=coverage, wape_by_category=seg, mase_scale=scale,
                  n_train=len(tr), n_val=len(va), n_test=len(te), seed=seed)
    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, config.MODELS_DIR / "lag_forecaster.joblib")
        out.to_csv(config.DATA_PROCESSED / "lag_test_predictions.csv.gz", index=False)
    return bundle


# ------------------------------------------------------------------ A2: cold-start launch demand model
def product_level_errors(df, pred, weeks=26):
    d = df.assign(_p=np.asarray(pred))
    d = d[d["age_weeks"] < weeks]
    g = d.groupby("product_id").agg(actual=("units", "sum"), pred=("_p", "sum"), category=("category", "first"),
                                    n_weeks=("age_weeks", "nunique"))
    g = g[g.n_weeks >= weeks]
    g["log_ratio"] = np.log(g["actual"].clip(lower=1) / g["pred"].clip(lower=1))
    g["ape"] = (g["actual"] - g["pred"]).abs() / g["actual"].clip(lower=1)
    return g.reset_index()

def _oof(models_factory, name, X, y, groups, n_splits=5):
    oof = np.zeros(len(y))
    for tri, tei in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = models_factory()[name] if name != "Baseline" else GroupMeanBaseline()
        m.fit(X.iloc[tri], y.iloc[tri])
        oof[tei] = m.predict(X.iloc[tei])
    return np.clip(oof, 0, None)

def train_coldstart_model(panel: pd.DataFrame, seed=config.SEED, save=True) -> dict | None:
    ref_all = fit_category_reference(panel)
    cs = build_coldstart_frame(panel, ref_all)
    n_prod = cs["product_id"].nunique() if len(cs) else 0
    if n_prod < 20:
        return None  # not enough observed launches -> caller must explain
    rng = np.random.default_rng(seed)
    prods = np.array(sorted(cs.product_id.unique()))
    test_prods = set(rng.choice(prods, max(4, int(0.2 * len(prods))), replace=False))
    dev = cs[~cs.product_id.isin(test_prods)].reset_index(drop=True)
    tst = cs[cs.product_id.isin(test_prods)].reset_index(drop=True)
    ref = fit_category_reference(dev)
    dev, tst = build_coldstart_frame(dev, ref), build_coldstart_frame(tst, ref)
    feats = CS_CAT + CS_NUM
    Xd, yd, Xt, yt = dev[feats], dev["units"], tst[feats], tst["units"]
    res = {"cv_dev": {}, "test": {}}
    names = ["Baseline"] + list(cs_candidates(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in names:
            oof = _oof(lambda: cs_candidates(seed), n, Xd, yd, dev["product_id"])
            pl = product_level_errors(dev, oof)
            res["cv_dev"][n] = {**forecast_metrics(yd, oof), "product_MdAPE_26w": float(pl["ape"].median()) if len(pl) else None}
        sel = min([n for n in names if n != "Baseline"], key=lambda n: res["cv_dev"][n]["WAPE"])
        fitted = {}
        for n in names:
            m = GroupMeanBaseline() if n == "Baseline" else cs_candidates(seed)[n]
            m.fit(Xd, yd)
            fitted[n] = m
            pt = np.clip(m.predict(Xt), 0, None)
            pl = product_level_errors(tst, pt)
            res["test"][n + (" [selected]" if n == sel else "")] = {**forecast_metrics(yt, pt),
                "product_MdAPE_26w": float(pl["ape"].median()) if len(pl) else None, "n_products": int(len(pl))}
        pt_sel = np.clip(fitted[sel].predict(Xt), 0, None)
        sub = tst.sample(min(8000, len(tst)), random_state=seed)
        pi = permutation_importance(fitted[sel], sub[feats], sub["units"], scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=seed)
        importance = sorted(zip(feats, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])
        # --- uncertainty calibration on ALL launch-cohort products (out-of-fold) ---
        full = build_coldstart_frame(panel, ref_all)
        Xf, yf = full[feats], full["units"]
        oof_all = _oof(lambda: cs_candidates(seed), sel, Xf, yf, full["product_id"])
        pl_all = product_level_errors(full, oof_all)
        d = full.assign(_p=oof_all)
        d = d[d.age_weeks < 26]
        pw = d.groupby(["product_id", "date"]).agg(a=("units", "sum"), p=("_p", "sum")).reset_index()
        pw["lr"] = np.log(pw.a.clip(lower=1) / pw.p.clip(lower=1))
        pw["lr_c"] = pw.lr - pw.groupby("product_id").lr.transform("mean")
        weekly_sd = float(pw["lr_c"].std())
        by_cat_err = pl_all.groupby("category")["ape"].median().to_dict()
        final = cs_candidates(seed)[sel]
        final.fit(Xf, yf)
    support = dict(
        price_index_range=(float(full.price_index.quantile(0.01)), float(full.price_index.quantile(0.99))),
        pack_size_range=(float(np.exp(full.log_pack_size.min())), float(np.exp(full.log_pack_size.max()))),
        categories=sorted(full.category.unique()), regions=sorted(full.region.unique()),
        store_types=sorted(full.store_type.unique()),
        products_per_category=full.groupby("category").product_id.nunique().to_dict(),
        n_products=int(full.product_id.nunique()), n_rows=int(len(full)),
        discount_range=(0.0, float(full.discount_pct.max())), age_range=(0, config.COLD_START_WEEKS - 1))
    bundle = dict(kind="coldstart", model=final, model_name=sel, features=feats, ref=ref_all, support=support,
                  log_ratio=pl_all["log_ratio"].to_numpy(), n_ratio_products=int(len(pl_all)),
                  weekly_noise_sd=weekly_sd, metrics=res, importance=importance,
                  product_ape_by_category=by_cat_err, product_errors=pl_all[["product_id", "category", "actual", "pred", "log_ratio", "ape"]],
                  test_products=sorted(test_prods), seed=seed,
                  oof_residuals=pd.DataFrame({"units": yf.values, "pred": oof_all, "category": full.category.values,
                                              "age_weeks": full.age_weeks.values}).sample(min(20000, len(full)), random_state=seed))
    if save:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, config.MODELS_DIR / "coldstart_model.joblib")
    return bundle


def load_bundle(name: str):
    p = config.MODELS_DIR / f"{name}.joblib"
    return joblib.load(p) if p.exists() else None

def models_trained() -> bool:
    return (config.MODELS_DIR / "coldstart_model.joblib").exists()
