"""Competitive and market intelligence.

No live scraping is performed. Two evidence types are kept strictly separate:
  * dataset reference prices (HISTORICAL, as of the dataset end date - never labelled current)
  * user-provided competitor prices (CSV / manual entry, with source, URL and retrieval timestamp)
"""
from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data_ingestion import to_base_size

def normalise_ppu100(price, size, unit):
    base, _ = to_base_size(size, unit)
    return price / base * 100.0

def freshness_status(ts, now=None) -> str:
    now = now or datetime.now(timezone.utc)
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t): return "unknown"
    age = (now - t).days
    return "fresh (<=30d)" if age <= 30 else "ageing (31-90d)" if age <= 90 else f"STALE ({age}d old)"

def catalog_reference(panel: pd.DataFrame, catalog: pd.DataFrame, weeks: int = 13) -> pd.DataFrame:
    end = panel.date.max()
    recent = panel[panel.date > end - pd.Timedelta(weeks=weeks)]
    p = recent.groupby("product_id").agg(price=("price", "median"), units=("units", "sum")).reset_index()
    c = catalog.merge(p, on="product_id")
    c["ppu100"] = [normalise_ppu100(a, b, u) for a, b, u in zip(c.price, c.pack_size, c.pack_unit)]
    c["price_basis"] = f"Dataset median price, last {weeks} weeks to {end.date()} (HISTORICAL, not current)"
    c["evidence"] = "dataset-observed"
    return c

def clean_competitors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["price"] = pd.to_numeric(d["price"]); d["pack_size"] = pd.to_numeric(d["pack_size"])
    d["ppu100"] = [normalise_ppu100(a, b, u) for a, b, u in zip(d.price, d.pack_size, d.pack_unit)]
    d["freshness"] = d["retrieved_at"].map(freshness_status)
    d["evidence"] = "user-provided"
    return d

def combined_view(ref: pd.DataFrame, comps: pd.DataFrame | None, category: str) -> pd.DataFrame:
    a = ref[ref.category == category][["product_name", "brand", "category", "pack_size", "pack_unit", "price", "ppu100", "evidence", "price_basis"]]
    a = a.rename(columns={"price_basis": "basis"})
    if comps is not None and len(comps):
        b = comps[comps.category == category][["product_name", "brand", "category", "pack_size", "pack_unit", "price", "ppu100", "evidence", "freshness"]]
        b = b.rename(columns={"freshness": "basis"})
        a = pd.concat([a, b], ignore_index=True)
    return a

def positioning_summary(my_ppu100: float, view: pd.DataFrame, differentiation="medium") -> dict:
    if view.empty:
        return dict(available=False)
    s = np.sort(view.ppu100.to_numpy())
    pct = float((s < my_ppu100).mean() * 100)
    q = np.quantile(s, [0.25, 0.5, 0.75, 0.9])
    tier = "economy" if my_ppu100 < q[0] else "mass market" if my_ppu100 < q[2] else "premium" if my_ppu100 < q[3] else "super-premium"
    hi = {"low": q[1], "medium": q[2], "high": q[3]}.get(differentiation, q[2])
    gaps = [(s[i], s[i + 1]) for i in range(len(s) - 1)]
    big = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:3]
    return dict(available=True, percentile=pct, implied_tier=tier, median_ppu100=float(q[1]), q25=float(q[0]), q75=float(q[2]),
                suggested_low=float(q[0]), suggested_high=float(hi), index_vs_median=float(my_ppu100 / q[1]),
                largest_gaps=[(float(a), float(b)) for a, b in big], n_comparables=int(len(s)))

def pack_gaps(view: pd.DataFrame) -> list:
    sizes = sorted({to_base_size(a, u)[0] for a, u in zip(view.pack_size, view.pack_unit)})
    return sizes

def load_macro_csv(path):
    """Optional macro indicators (e.g. CPI) the user downloaded from MOSPI / World Bank. Columns:
    date, indicator, value, source_name, source_url, retrieved_at."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["freshness"] = df["retrieved_at"].map(freshness_status)
    return df
