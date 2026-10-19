"""Schema validation, data-quality reporting and launch-input validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config

def validate_schema(df: pd.DataFrame, required: list[str]) -> list[str]:
    errs = []
    if df is None or len(df) == 0:
        return ["Dataset is empty."]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errs.append(f"Missing required columns: {missing}")
    return errs

def validate_panel(df: pd.DataFrame) -> dict:
    """Return {'errors': [...], 'warnings': [...]} for a panel dataframe."""
    errors = validate_schema(df, config.REQUIRED_PANEL_COLS)
    warnings = []
    if errors:
        return dict(errors=errors, warnings=warnings)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        errors.append("Column 'date' is not a datetime.")
    if (df["units"] < 0).any():
        errors.append(f"{int((df['units'] < 0).sum())} rows have negative units.")
    if (df["price"] <= 0).any():
        errors.append(f"{int((df['price'] <= 0).sum())} rows have non-positive price.")
    dup = df.duplicated(["date", "product_id", "store_id"]).sum()
    if dup:
        errors.append(f"{int(dup)} duplicate (date, product_id, store_id) rows.")
    if df.groupby(["product_id", "store_id"]).size().median() < 30:
        warnings.append("Median series length < 30 weeks: limited history for lag features.")
    miss = df[config.REQUIRED_PANEL_COLS].isna().mean()
    for c, v in miss[miss > 0].items():
        warnings.append(f"Column '{c}' has {v:.1%} missing values.")
    return dict(errors=errors, warnings=warnings)

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        r = dict(column=c, dtype=str(s.dtype), missing=int(s.isna().sum()), missing_pct=round(100 * s.isna().mean(), 2),
                 unique=int(s.nunique()))
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.notna().any():
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            r["outliers_iqr"] = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
            r["min"], r["max"] = float(s.min()), float(s.max())
        rows.append(r)
    return pd.DataFrame(rows)

def validate_launch_inputs(d: dict) -> dict:
    """Validate a LaunchInputs dict. Returns {'errors': [...], 'warnings': [...]}."""
    e, w = [], []
    def num(k, lo=None, hi=None, req=True):
        v = d.get(k)
        if v is None:
            if req: e.append(f"'{k}' is required.")
            return
        if lo is not None and v < lo: e.append(f"'{k}' must be >= {lo} (got {v}).")
        if hi is not None and v > hi: e.append(f"'{k}' must be <= {hi} (got {v}).")
    if not str(d.get("product_name", "")).strip(): e.append("Product name is required.")
    if not d.get("category"): e.append("Select a product category.")
    num("price", 0.01); num("pack_size", 0.01)
    for k in ("mfg_cost", "packaging_cost", "dist_cost", "ad_budget", "digital_budget", "retail_activation_budget",
              "trade_promo_budget", "influencer_budget", "other_fixed_costs", "initial_inventory", "weekly_capacity"):
        num(k, 0)
    for k in ("tax_rate", "trade_margin", "intro_discount", "discount_funded_share", "wastage_pct", "availability"):
        num(k, 0, 1)
    if d.get("intro_discount", 0) > 0.5: e.append("Introductory discount above 50% is not supported.")
    num("lead_time_weeks", 0, 26); num("campaign_weeks", 0, 52); num("intro_discount_weeks", 0, 52)
    if not d.get("regions"): e.append("Select at least one target region.")
    if sum((d.get("outlets") or {}).values()) <= 0: e.append("Plan at least one retail outlet.")
    if any(v < 0 for v in (d.get("outlets") or {}).values()): e.append("Outlet counts cannot be negative.")
    if d.get("pack_unit") not in config.PACK_UNITS: e.append(f"Pack unit must be one of {config.PACK_UNITS}.")
    if d.get("currency", config.CURRENCY) != config.CURRENCY:
        e.append(f"Inconsistent currency: model data are in {config.CURRENCY}.")
    ld = pd.to_datetime(d.get("launch_date"), errors="coerce")
    if pd.isna(ld): e.append("Launch date is invalid.")
    # economics
    if not e:
        net_rev = d["price"] / (1 + d["tax_rate"]) * (1 - d["trade_margin"])
        vc = d["mfg_cost"] + d["packaging_cost"] + d["dist_cost"]
        if vc >= net_rev:
            w.append(f"Variable cost/unit ({vc:.2f}) is >= manufacturer net revenue/unit ({net_rev:.2f}); "
                     "every unit sold loses money before launch spend.")
        total = sum(d.get(k, 0) for k in ("ad_budget", "digital_budget", "retail_activation_budget",
                                         "trade_promo_budget", "influencer_budget"))
        if d.get("total_launch_budget") and total > d["total_launch_budget"]:
            e.append(f"Budget lines ({total:,.0f}) exceed the total launch budget ({d['total_launch_budget']:,.0f}).")
        if d.get("competitor_price_low") and d.get("competitor_price_high") and d["competitor_price_low"] > d["competitor_price_high"]:
            e.append("Competitor price range: low is greater than high.")
        if d.get("initial_inventory", 0) == 0 and d.get("weekly_capacity", 0) == 0:
            e.append("Both initial inventory and weekly capacity are zero: nothing can be sold.")
    return dict(errors=e, warnings=w)

def validate_competitor_upload(df: pd.DataFrame) -> dict:
    req = ["product_name", "brand", "category", "pack_size", "pack_unit", "price", "currency",
           "price_type", "source_name", "source_url", "retrieved_at", "market"]
    errs = validate_schema(df, req)
    if errs:
        return dict(errors=errs, warnings=[])
    w = []
    if (pd.to_numeric(df["price"], errors="coerce") <= 0).any() or pd.to_numeric(df["price"], errors="coerce").isna().any():
        errs.append("Some prices are non-numeric or non-positive.")
    if (pd.to_numeric(df["pack_size"], errors="coerce") <= 0).any() or pd.to_numeric(df["pack_size"], errors="coerce").isna().any():
        errs.append("Some pack sizes are non-numeric or non-positive.")
    bad = ~df["pack_unit"].astype(str).str.lower().isin([u.lower() for u in config.PACK_UNITS])
    if bad.any(): errs.append(f"Unsupported pack_unit values: {sorted(df.loc[bad, 'pack_unit'].astype(str).unique())}")
    if pd.to_datetime(df["retrieved_at"], errors="coerce").isna().any():
        errs.append("Some retrieved_at timestamps are invalid.")
    if (df["currency"].astype(str) != config.CURRENCY).any():
        errs.append(f"Currency must be {config.CURRENCY} for all rows (no silent conversion).")
    bad = ~df["price_type"].astype(str).str.lower().isin(["regular", "discounted", "promotional"])
    if bad.any(): errs.append("price_type must be regular, discounted or promotional.")
    return dict(errors=errs, warnings=w)
