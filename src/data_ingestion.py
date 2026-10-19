"""Data loading, dataset-mode detection, and adapters for real public datasets.

Real-data path: scripts/download_data.py fetches Online Retail II (UCI) -> adapt_online_retail_ii()
-> standard panel/transactions schema. The adapter was exercised only on a format fixture inside the
build sandbox (network access to UCI was blocked there); validate it on your machine.
"""
from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from . import config

# ---------------------------------------------------------------- loading
def dataset_meta() -> dict:
    if config.META_PATH.exists():
        return json.loads(config.META_PATH.read_text())
    return {}

def is_demo() -> bool:
    return dataset_meta().get("mode", "demo_synthetic") != "real"

def data_available() -> bool:
    return config.PANEL_PATH.exists() and config.CATALOG_PATH.exists()

def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.PANEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Panel dataset not found at {path}. Run `python scripts/validate_data.py` "
                                "(creates the labelled demo dataset) or import a real dataset first.")
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ("promo_flag", "holiday_flag"):
        if c in df:
            df[c] = df[c].fillna(0).astype(int)
    return df

def load_catalog(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or config.CATALOG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found at {path}.")
    return pd.read_csv(path, parse_dates=["launch_date"])

def load_transactions(path: Path | None = None) -> pd.DataFrame | None:
    path = Path(path or config.TRANSACTIONS_PATH)
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["invoice_date"])

# ---------------------------------------------------------------- provenance
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def record_provenance(name: str, url: str, path: Path, licence: str, notes: str = "") -> dict:
    rec = dict(name=name, url=url, file=str(path), sha256=sha256_of(path), size_bytes=path.stat().st_size,
               accessed_at=datetime.now(timezone.utc).isoformat(), licence=licence, notes=notes)
    log = config.DATA_RAW / "provenance.json"
    items = json.loads(log.read_text()) if log.exists() else []
    items.append(rec)
    log.write_text(json.dumps(items, indent=2))
    return rec

# ---------------------------------------------------------------- unit helpers
def to_base_size(size, unit):
    """Convert pack size to g / ml / unit. Returns (size_in_base, base_unit)."""
    u = str(unit).lower()
    if u == "kg": return size * 1000, "g"
    if u == "l": return size * 1000, "ml"
    return size, u

# ---------------------------------------------------------------- Online Retail II adapter
_NON_PRODUCT = {"POST", "D", "DOT", "M", "BANK CHARGES", "CRUK", "C2", "AMAZONFEE", "S", "B", "PADS", "ADJUST", "TEST001"}
_CATEGORY_RULES = [
    ("Seasonal & Gifting", ["CHRISTMAS", "XMAS", "EASTER", "ADVENT", "GIFT", "SANTA", "WRAP", "CARD"]),
    ("Kitchen & Tableware", ["MUG", "CUP", "PLATE", "BOWL", "TEA", "JAR", "CAKE", "SPOON", "KITCHEN", "BAKING", "LUNCH"]),
    ("Home Decor", ["CANDLE", "LIGHT", "LAMP", "FRAME", "CUSHION", "DECOR", "HEART", "SIGN", "CLOCK", "VASE"]),
    ("Stationery & Craft", ["NOTEBOOK", "PENCIL", "PEN ", "PAPER", "STICKER", "CRAFT", "RIBBON", "BOOK"]),
    ("Bags & Accessories", ["BAG", "PURSE", "WALLET", "UMBRELLA", "BOTTLE"]),
]

def derive_category(desc: str) -> str:
    d = str(desc).upper()
    for cat, kws in _CATEGORY_RULES:
        if any(k in d for k in kws):
            return cat
    return "Other"

def adapt_online_retail_ii(raw: pd.DataFrame, min_weeks: int = 20):
    """Map Online Retail II columns -> standard (panel, catalog, transactions).

    Caveats (also surfaced in the UI): UK gift wholesaler (not FMCG); prices in GBP; categories are
    keyword-derived from descriptions; 'store' = customer country; no promotions or holidays;
    launch dates are PROXIED by first-sale week (left-censored for products sold from the start).
    """
    df = raw.rename(columns={"Invoice": "invoice_id", "StockCode": "product_id", "Description": "desc",
                             "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "unit_price",
                             "Customer ID": "customer_id", "Country": "country"}).copy()
    need = {"invoice_id", "product_id", "quantity", "invoice_date", "unit_price", "country"}
    if not need <= set(df.columns):
        raise ValueError(f"Missing Online Retail II columns: {sorted(need - set(df.columns))}")
    df["invoice_id"] = df["invoice_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str).str.upper()
    df = df[~df.invoice_id.str.startswith("C")]
    df = df[(df.quantity > 0) & (df.unit_price > 0) & ~df.product_id.isin(_NON_PRODUCT)].copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["category"] = df["desc"].map(derive_category)
    tx = df[["invoice_id", "customer_id", "invoice_date", "product_id", "category", "quantity", "unit_price"]].dropna(subset=["customer_id"]).copy()
    tx["customer_id"] = tx["customer_id"].astype(int).astype(str)
    tx["discount_pct"] = 0.0
    df["date"] = df["invoice_date"].dt.to_period("W-SUN").dt.start_time
    g = (df.assign(region=df.country, store_id=df.country)
         .groupby(["date", "product_id", "store_id", "category", "region"], as_index=False)
         .agg(units=("quantity", "sum"), price=("unit_price", "median")))
    g["store_type"] = "Wholesale"
    g["promo_flag"] = 0
    g["discount_pct"] = 0.0
    g["holiday_flag"] = g["date"].dt.isocalendar().week.astype(int).isin(config.HOLIDAY_WEEKS).astype(int)
    g = complete_weekly(g, ["product_id", "store_id"])
    first = g[g.units > 0].groupby("product_id").date.min().rename("launch_date")
    t0 = g.date.min()
    cat = (g.groupby("product_id").agg(category=("category", "first")).join(first).reset_index())
    cat["launch_observed"] = cat.launch_date > t0 + pd.Timedelta(weeks=8)
    cat["product_name"] = cat.product_id
    g = g.merge(cat[["product_id", "launch_date", "launch_observed"]], on="product_id")
    g = g[g.date >= g.launch_date]
    g["weeks_since_intro"] = np.where(g.launch_observed, (g.date - g.launch_date).dt.days / 7, np.nan)
    g = g.drop(columns="launch_date")
    keep = g.groupby("product_id").date.nunique()
    g = g[g.product_id.isin(keep[keep >= min_weeks].index)].reset_index(drop=True)
    return g, cat[cat.product_id.isin(g.product_id.unique())].reset_index(drop=True), tx

def complete_weekly(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Fill missing weeks between each series' first and last observation with zero units."""
    out = []
    for k, grp in df.groupby(keys, sort=False):
        idx = pd.date_range(grp.date.min(), grp.date.max(), freq="7D")
        grp = grp.set_index("date").reindex(idx)
        grp.index.name = "date"
        grp["units"] = grp["units"].fillna(0)
        for c in grp.columns:
            if c not in ("units", "promo_flag", "discount_pct", "holiday_flag"):
                grp[c] = grp[c].ffill().bfill()
        for c in ("promo_flag", "discount_pct"):
            grp[c] = grp[c].fillna(0)
        grp["holiday_flag"] = grp.index.isocalendar().week.astype(int).isin(config.HOLIDAY_WEEKS).astype(int).values
        out.append(grp.reset_index())
    return pd.concat(out, ignore_index=True) if out else df

def import_real_dataset(panel, catalog, tx, name, source_url, licence, currency="GBP"):
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.PANEL_PATH, index=False)
    catalog.to_csv(config.CATALOG_PATH, index=False)
    if tx is not None:
        tx.to_csv(config.TRANSACTIONS_PATH, index=False)
    meta = dict(mode="real", name=name, source_url=source_url, licence=licence, currency=currency,
                imported_at=datetime.now(timezone.utc).isoformat(), panel_rows=len(panel),
                start=str(panel.date.min().date()), end=str(panel.date.max().date()),
                n_products=int(catalog.shape[0]), has_launch_labels=False,
                note="Real transactions; launch outcomes NOT observed; launch dates proxied by first sale.")
    config.META_PATH.write_text(json.dumps(meta, indent=2))
    return meta
