"""Leakage-safe feature engineering.

Lag-model rule: for a direct h-week-ahead forecast of week t, every history-derived feature is built
from information dated <= t-h (all series are shifted by h BEFORE any rolling statistic).
Price/promo/holiday for week t are treated as *planned/known* covariates (documented assumption).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .data_ingestion import to_base_size

CS_CAT = ["category", "region", "store_type"]
CS_NUM = ["log_price_index", "log_pack_size", "promo_flag", "discount_pct", "holiday_flag",
          "woy_sin", "woy_cos", "age_weeks"]
LAG_CAT = ["category", "region", "store_type"]
LAG_NUM = ["lag_h", "lag_h1", "lag_h4", "lag_52", "roll_mean_4", "roll_mean_8", "roll_std_8", "ewm_03",
           "price_ratio_recent", "log_price_index", "promo_flag", "discount_pct", "holiday_flag",
           "woy_sin", "woy_cos", "series_age"]


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    woy = d.dt.isocalendar().week.astype(int)
    return pd.DataFrame({"woy": woy.values, "month": d.dt.month.values,
                         "woy_sin": np.sin(2 * np.pi * woy.values / 52), "woy_cos": np.cos(2 * np.pi * woy.values / 52)},
                        index=dates.index)


def _pack(df: pd.DataFrame):
    size = df["pack_size"] if "pack_size" in df else pd.Series(1.0, index=df.index)
    unit = df["pack_unit"] if "pack_unit" in df else pd.Series("unit", index=df.index)
    size = size.fillna(1.0)
    unit = unit.fillna("unit")
    base = [to_base_size(s, u)[0] for s, u in zip(size, unit)]
    return pd.Series(base, index=df.index, dtype=float)


def ppu100(price, base_size):
    """Price per 100 base units (g / ml / piece)."""
    return price / base_size * 100.0


def fit_category_reference(panel: pd.DataFrame) -> dict:
    """Median (over products) of each product's median price-per-100 units, by category."""
    d = panel.assign(_bs=_pack(panel))
    d["_ppu"] = ppu100(d["price"], d["_bs"])
    per_prod = d.groupby(["category", "product_id"])["_ppu"].median().reset_index()
    return per_prod.groupby("category")["_ppu"].median().to_dict()


def add_price_index(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    bs = _pack(df)
    df["log_pack_size"] = np.log(bs.clip(lower=1e-6))
    df["ppu100"] = ppu100(df["price"], bs)
    r = df["category"].map(ref)
    df["price_index"] = df["ppu100"] / r
    df["log_price_index"] = np.log(df["price_index"].clip(lower=1e-6))
    return df


def build_lag_features(panel: pd.DataFrame, ref: dict, h: int = config.HORIZON_WEEKS) -> pd.DataFrame:
    df = panel.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)
    keys = ["product_id", "store_id"]
    gaps = df.groupby(keys)["date"].diff().dropna()
    if len(gaps) and (gaps != pd.Timedelta(days=7)).any():
        raise ValueError("Series are not contiguous weekly; run complete_weekly() before building lag features.")
    g = df.groupby(keys, sort=False)
    df["lag_h"] = g["units"].shift(h)
    df["lag_h1"] = g["units"].shift(h + 1)
    df["lag_h4"] = g["units"].shift(h + 4)
    df["lag_52"] = g["units"].shift(52)
    s = df["lag_h"]
    sg = s.groupby([df["product_id"], df["store_id"]], sort=False)
    df["roll_mean_4"] = sg.transform(lambda x: x.rolling(4, min_periods=2).mean())
    df["roll_mean_8"] = sg.transform(lambda x: x.rolling(8, min_periods=3).mean())
    df["roll_std_8"] = sg.transform(lambda x: x.rolling(8, min_periods=3).std())
    df["ewm_03"] = sg.transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    ps = g["price"].shift(h).groupby([df["product_id"], df["store_id"]], sort=False)
    df["price_ratio_recent"] = df["price"] / ps.transform(lambda x: x.rolling(8, min_periods=3).mean())
    df["series_age"] = g.cumcount().astype(float)
    df = add_price_index(df, ref)
    df = pd.concat([df, calendar_features(df["date"])], axis=1)
    for c, v in (("promo_flag", 0), ("discount_pct", 0), ("holiday_flag", 0)):
        if c not in df:
            df[c] = v
    return df


def build_coldstart_frame(panel: pd.DataFrame, ref: dict, weeks: int = config.COLD_START_WEEKS) -> pd.DataFrame:
    """Launch-cohort rows only: products whose introduction was observed, first `weeks` weeks of life."""
    if "launch_observed" not in panel or "weeks_since_intro" not in panel:
        return panel.iloc[0:0].copy()
    df = panel[panel["launch_observed"].astype(bool) & (panel["weeks_since_intro"] < weeks)].copy()
    derived = ["log_pack_size", "ppu100", "price_index", "log_price_index", "woy", "month", "woy_sin", "woy_cos", "age_weeks"]
    df = df.drop(columns=[c for c in derived if c in df.columns])
    df["age_weeks"] = df["weeks_since_intro"]
    df = add_price_index(df, ref)
    df = pd.concat([df, calendar_features(df["date"])], axis=1)
    for c in ("promo_flag", "discount_pct", "holiday_flag"):
        if c not in df:
            df[c] = 0
    return df
