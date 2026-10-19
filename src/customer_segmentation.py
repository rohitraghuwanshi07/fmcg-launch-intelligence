"""Model D: RFM + K-Means segmentation, repeat purchase and basket analysis (needs customer-level data)."""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from . import config

def build_rfm(tx: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    if tx is None or tx.empty or tx["customer_id"].isna().all():
        raise ValueError("Customer-level transactions with customer_id are required for segmentation.")
    t = tx.dropna(subset=["customer_id"]).copy()
    t["value"] = t["quantity"] * t["unit_price"]
    snapshot = pd.Timestamp(snapshot) if snapshot is not None else t["invoice_date"].max() + pd.Timedelta(days=1)
    g = t.groupby("customer_id")
    r = pd.DataFrame({
        "recency_days": (snapshot - g["invoice_date"].max()).dt.days,
        "frequency": g["invoice_id"].nunique(),
        "monetary": g["value"].sum(),
        "n_categories": g["category"].nunique() if "category" in t else 1,
    })
    r["avg_order_value"] = r["monetary"] / r["frequency"]
    r["discount_share"] = g["discount_pct"].apply(lambda s: float((s > 0).mean())) if "discount_pct" in t else 0.0
    return r.reset_index()

def segment_customers(rfm: pd.DataFrame, k: int | None = None, k_range=range(2, 8), seed=config.SEED):
    if len(rfm) < 50:
        raise ValueError("Need at least 50 customers for segmentation.")
    X = np.column_stack([np.log1p(rfm.recency_days), np.log1p(rfm.frequency), np.log1p(rfm.monetary), rfm.discount_share])
    Xs = StandardScaler().fit_transform(X)
    sil = {}
    sample = np.random.default_rng(seed).choice(len(Xs), min(2000, len(Xs)), replace=False)
    for kk in k_range:
        lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(Xs)
        sil[kk] = float(silhouette_score(Xs[sample], lab[sample]))
    best_k = k or max(sil, key=sil.get)
    km = KMeans(best_k, n_init=10, random_state=seed).fit(Xs)
    out = rfm.copy()
    out["cluster"] = km.labels_
    prof = out.groupby("cluster").agg(customers=("customer_id", "size"), recency_days=("recency_days", "mean"),
                                      frequency=("frequency", "mean"), monetary=("monetary", "mean"),
                                      avg_order_value=("avg_order_value", "mean"), discount_share=("discount_share", "mean"),
                                      n_categories=("n_categories", "mean"))
    prof["share_pct"] = 100 * prof.customers / prof.customers.sum()
    prof["segment"] = _name_segments(prof)
    out["segment"] = out.cluster.map(prof["segment"])
    return out, prof.reset_index(), dict(silhouette_by_k=sil, chosen_k=int(best_k), silhouette=sil.get(best_k))

def _name_segments(p: pd.DataFrame) -> pd.Series:
    """Names are assigned from measured cluster characteristics (rule-based), never invented demographics."""
    names, left = {}, set(p.index)
    def pick(label, score):
        if not left: return
        c = score[list(left)].idxmax(); names[c] = label; left.discard(c)
    pick("Lapsed / at-risk", p.recency_days)
    pick("Loyal high-value", p.frequency.rank() + p.monetary.rank())
    pick("Deal-seekers", p.discount_share)
    pick("Frequent small-basket", p.frequency)
    for c in left: names[c] = f"Occasional shoppers ({c})"
    return pd.Series(names)

def category_mix_by_segment(tx: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    t = tx.merge(seg[["customer_id", "segment"]], on="customer_id")
    t["value"] = t.quantity * t.unit_price
    m = t.pivot_table(index="segment", columns="category", values="value", aggfunc="sum", fill_value=0)
    return (m.div(m.sum(axis=1), axis=0) * 100).round(1)

def repeat_purchase(tx: pd.DataFrame) -> pd.DataFrame:
    g = tx.groupby(["category", "customer_id"])["invoice_id"].nunique().reset_index()
    return g.groupby("category").agg(buyers=("customer_id", "size"),
                                     repeat_rate_pct=("invoice_id", lambda s: 100 * (s >= 2).mean())).reset_index()

def basket_pairs(tx: pd.DataFrame, level="category", min_support=0.005, top=15, max_items=12) -> pd.DataFrame:
    baskets = tx.groupby("invoice_id")[level].apply(lambda s: sorted(set(s)))
    n = len(baskets)
    single, pair = {}, {}
    for items in baskets:
        for i in items: single[i] = single.get(i, 0) + 1
        if len(items) <= max_items:
            for a, b in combinations(items, 2): pair[(a, b)] = pair.get((a, b), 0) + 1
    rows = [dict(item_a=a, item_b=b, support=c / n, confidence_a_to_b=c / single[a],
                 lift=(c / n) / ((single[a] / n) * (single[b] / n))) for (a, b), c in pair.items() if c / n >= min_support]
    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(top).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["item_a", "item_b", "support", "confidence_a_to_b", "lift"])
