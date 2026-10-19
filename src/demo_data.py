"""Synthetic DEMO dataset generator.

EVERYTHING produced here is SIMULATED. It exists so the platform runs end-to-end when no real
dataset is available. It is generated from a documented data-generating process (DGP) with known
parameters, which lets the test-suite verify that estimators recover planted effects. Results
computed on it must never be presented as real-world observations.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from . import config

CATEGORY_SPECS = {
    "Biscuits & Snacks": dict(subs=["Biscuits", "Namkeen", "Chips"], unit="g", sizes=[50, 100, 200, 400],
                              ppu100=35, base=55, elast=-1.9, amp=0.10, peak=44, promo_up=0.22, trend=0.06, infl=0.05, hol=0.25),
    "Beverages & Juices": dict(subs=["Fruit Juice", "Soft Drinks", "Energy Drinks"], unit="ml", sizes=[200, 500, 1000],
                               ppu100=12, base=70, elast=-2.1, amp=0.30, peak=22, promo_up=0.28, trend=0.08, infl=0.04, hol=0.10),
    "Personal Care": dict(subs=["Shampoo", "Body Wash", "Face Wash"], unit="ml", sizes=[50, 100, 200],
                          ppu100=120, base=30, elast=-1.3, amp=0.08, peak=47, promo_up=0.15, trend=0.05, infl=0.04, hol=0.08),
    "Home Care": dict(subs=["Dish Wash", "Floor Cleaner", "Laundry"], unit="ml", sizes=[250, 500, 1000],
                      ppu100=20, base=28, elast=-1.5, amp=0.05, peak=10, promo_up=0.18, trend=0.04, infl=0.04, hol=0.10),
    "Packaged Grocery": dict(subs=["Noodles", "Spreads", "Ready Mixes"], unit="g", sizes=[100, 250, 500, 1000],
                             ppu100=18, base=45, elast=-1.7, amp=0.10, peak=30, promo_up=0.20, trend=0.05, infl=0.05, hol=0.12),
    "Dairy & Related": dict(subs=["Flavoured Milk", "Curd", "Paneer"], unit="ml", sizes=[200, 500, 1000],
                            ppu100=25, base=60, elast=-1.2, amp=0.06, peak=18, promo_up=0.12, trend=0.03, infl=0.03, hol=0.06),
}
BRANDS = ["Aarav Foods", "Nirmal", "GreenLeaf", "Sunrise", "Kisan Fresh", "PureLife", "Zesto", "HomeGlow"]  # fictional
TIER_PRICE = {"economy": 0.8, "mass market": 1.0, "premium": 1.4}
TIER_VOL = {"economy": 0.25, "mass market": 0.0, "premium": -0.35}
REGIONS = {"North": 0.0, "South": 0.10, "East": -0.15, "West": 0.15}
STORE_TYPES = {"Hypermarket": 0.5, "Supermarket": 0.0, "Convenience": -0.6}
PLANTED_ELASTICITY = {k: v["elast"] for k, v in CATEGORY_SPECS.items()}


def make_stores() -> pd.DataFrame:
    rows, n = [], 0
    for r in REGIONS:
        for st in STORE_TYPES:
            n += 1
            rows.append(dict(store_id=f"S{n:02d}", region=r, store_type=st))
    return pd.DataFrame(rows)


def generate_demo_panel(seed: int = config.SEED, n_products: int = 72, n_weeks: int = 104,
                        start: str = "2023-01-02", n_established: int = 12):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    stores = make_stores()
    cats = list(CATEGORY_SPECS)
    established = set(rng.choice(n_products, n_established, replace=False).tolist())
    cat_region = {(c, r): rng.normal(0, 0.10) for c in cats for r in REGIONS}
    catalog, frames = [], []
    for i in range(n_products):
        cat = cats[i % len(cats)]
        sp = CATEGORY_SPECS[cat]
        sub = sp["subs"][rng.integers(len(sp["subs"]))]
        size = int(rng.choice(sp["sizes"]))
        tier = str(rng.choice(list(TIER_PRICE), p=[0.25, 0.5, 0.25]))
        brand = BRANDS[rng.integers(len(BRANDS))]
        pid = f"P{i + 1:03d}"
        base_price = max(1.0, round(sp["ppu100"] * size / 100 * TIER_PRICE[tier] * rng.lognormal(0, 0.05)))
        prod_eff, tau = rng.normal(0, 0.35), rng.uniform(4, 14)
        est = i in established
        launch_week = 0 if est else int(rng.integers(1, 71))
        cov = rng.uniform(0.5, 1.0)
        n_st = max(4, int(round(cov * len(stores))))
        st_idx = rng.choice(len(stores), n_st, replace=False)
        ref_size = float(np.median(sp["sizes"]))
        catalog.append(dict(product_id=pid, product_name=f"{brand} {sub} {size}{sp['unit']}", category=cat,
                            subcategory=sub, brand=brand, pack_size=size, pack_unit=sp["unit"], positioning=tier,
                            launch_observed=not est, launch_date=pd.NaT if est else weeks[launch_week]))
        t = np.arange(launch_week, n_weeks)
        woy = weeks[t].isocalendar().week.to_numpy().astype(int)
        infl_f = (1 + sp["infl"]) ** (t / 52)
        prod_shock = np.exp(rng.normal(0, 0.025, len(t)))
        seas = sp["amp"] * np.cos(2 * np.pi * (woy - sp["peak"]) / 52)
        hol = np.isin(woy, list(config.HOLIDAY_WEEKS)).astype(int)
        age = t - launch_week
        ramp = np.ones(len(t)) if est else 1 - np.exp(-(age + 1) / tau)
        for si in st_idx:
            s = stores.iloc[si]
            shock = np.exp(rng.normal(0, 0.03, len(t)))
            promo = rng.random(len(t)) < 0.10
            disc = np.where(promo, rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], len(t)), 0.0)
            price = base_price * infl_f * prod_shock * shock * (1 - disc)
            rel = price / (base_price * infl_f)
            log_mu = (np.log(sp["base"]) + STORE_TYPES[s.store_type] + REGIONS[s.region] + cat_region[(cat, s.region)]
                      + prod_eff + TIER_VOL[tier] - 0.3 * np.log(size / ref_size) + sp["trend"] * t / 52 + seas
                      + sp["hol"] * hol + sp["elast"] * np.log(rel) + sp["promo_up"] * promo)
            mu = np.exp(log_mu) * ramp
            units = rng.poisson(rng.gamma(20, mu / 20))
            frames.append(pd.DataFrame(dict(
                date=weeks[t], product_id=pid, store_id=s.store_id, units=units, price=price.round(2),
                promo_flag=promo.astype(int), discount_pct=(disc * 100).round(0), holiday_flag=hol,
                weeks_since_intro=(np.full(len(t), np.nan) if est else age.astype(float)))))
    cat_df = pd.DataFrame(catalog)
    panel = (pd.concat(frames, ignore_index=True)
             .merge(cat_df.drop(columns=["product_name", "launch_date"]), on="product_id")
             .merge(stores, on="store_id"))
    return panel, cat_df


def generate_demo_transactions(catalog: pd.DataFrame, n_customers: int = 2500, seed: int = config.SEED):
    """Simulated customer transactions with 4 latent behavioural profiles."""
    rng = np.random.default_rng(seed + 1)
    start, end = pd.Timestamp("2023-01-02"), pd.Timestamp("2024-12-23")
    span = (end - start).days
    cats = list(CATEGORY_SPECS)
    prof = [dict(rate=14, basket=1.0, disc=0.05, churn=0.1), dict(rate=8, basket=3.5, disc=0.10, churn=0.15),
            dict(rate=6, basket=2.0, disc=0.55, churn=0.3), dict(rate=2, basket=1.5, disc=0.20, churn=0.5)]
    by_cat = {c: catalog[catalog.category == c].reset_index(drop=True) for c in cats}
    rows, inv = [], 0
    for cid in range(n_customers):
        p = prof[rng.integers(4)]
        prefs = rng.dirichlet(np.ones(len(cats)) * 0.8)
        last_day = span if rng.random() > p["churn"] else int(rng.uniform(0.2, 0.7) * span)
        nvis = max(1, rng.poisson(p["rate"] * last_day / 365))
        for d in np.sort(rng.integers(0, max(last_day, 1), nvis)):
            inv += 1
            for _ in range(1 + rng.poisson(p["basket"])):
                c = cats[rng.choice(len(cats), p=prefs)]
                pr = by_cat[c].iloc[rng.integers(len(by_cat[c]))]
                disc = int(rng.choice([10, 15, 20, 25])) if rng.random() < p["disc"] else 0
                price = pr.pack_size / 100 * CATEGORY_SPECS[c]["ppu100"] * TIER_PRICE[pr.positioning]
                rows.append((f"INV{inv:06d}", f"C{cid:05d}", start + pd.Timedelta(days=int(d)), pr.product_id, c,
                             1 + rng.poisson(0.6), round(price * (1 - disc / 100), 2), disc))
    return pd.DataFrame(rows, columns=["invoice_id", "customer_id", "invoice_date", "product_id", "category",
                                       "quantity", "unit_price", "discount_pct"])


def generate_and_save(out_dir=None, seed: int = config.SEED, n_products: int = 72, n_customers: int = 2500) -> dict:
    out_dir = Path(out_dir) if out_dir else config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, catalog = generate_demo_panel(seed, n_products=n_products)
    tx = generate_demo_transactions(catalog, n_customers=n_customers, seed=seed)
    panel.to_csv(out_dir / "panel.csv.gz", index=False)
    catalog.to_csv(out_dir / "catalog.csv", index=False)
    tx.to_csv(out_dir / "transactions.csv.gz", index=False)
    meta = dict(mode="demo_synthetic", name="Synthetic FMCG demo dataset (SIMULATED)",
                generated_at=datetime.now(timezone.utc).isoformat(), seed=seed,
                panel_rows=len(panel), transaction_rows=len(tx), n_products=int(catalog.shape[0]),
                start=str(panel.date.min().date()), end=str(panel.date.max().date()),
                currency=config.CURRENCY, has_launch_labels=False,
                note="All values are simulated. Not observed market data.")
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
