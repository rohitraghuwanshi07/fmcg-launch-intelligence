"""Model B: launch success.

A supervised launch-success classifier requires verified launch OUTCOME labels (e.g. 'reached the
contribution target within N weeks of launch') for a cohort of launches. Public sales datasets contain
sales of existing products, not verified launch outcomes, so this module refuses to fabricate labels.
The platform instead reports a commercial-hurdle attainment probability (scenario_engine.run_assessment).
"""
from __future__ import annotations
from . import config

REQUIRED_LABEL_COLUMNS = ["product_id", "launch_date", "category", "launch_outcome_label"]

def classifier_status(labels_df=None) -> dict:
    if labels_df is None:
        return dict(available=False, reason="No launch-outcome labels supplied. Sales history alone does not tell us "
                    "which products were successful launches.", needed=_needed())
    miss = [c for c in REQUIRED_LABEL_COLUMNS if c not in labels_df.columns]
    if miss:
        return dict(available=False, reason=f"Label file is missing columns {miss}.", needed=_needed())
    if len(labels_df) < 200 or labels_df["launch_outcome_label"].nunique() < 2 or labels_df["launch_outcome_label"].value_counts().min() < 30:
        return dict(available=False, reason=f"{len(labels_df)} labelled launches (min class < 30) is too few to train and "
                    "calibrate a classifier reliably.", needed=_needed())
    return dict(available=True, reason="Labels sufficient for a grouped, time-aware classifier (not implemented in demo mode).", needed=[])

def _needed():
    return ["200+ launches with verified outcomes (>=30 per class)", "launch date, category, price, pack, launch spend, distribution at launch",
            "outcome measured over a fixed window against an agreed hurdle", "features known at launch time only (no post-launch sales)",
            "validation grouped by brand/launch cohort with chronological ordering"]

def hurdle_definition() -> str:
    return config.HURDLE_RULE
