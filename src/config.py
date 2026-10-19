"""Central configuration: paths, constants, schema definitions, business rules."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = Path(os.environ.get("FMCG_PROCESSED_DIR", ROOT / "data" / "processed"))
DATA_REAL = ROOT / "data" / "real"          # REAL datasets bundled with the repo (see data/real/PROVENANCE.json)
DATA_EXTERNAL = ROOT / "data" / "external"
MODELS_DIR = Path(os.environ.get("FMCG_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
DB_PATH = Path(os.environ.get("FMCG_DB_PATH", ROOT / "data" / "scenarios.db"))

PANEL_PATH = DATA_PROCESSED / "panel.csv.gz"
CATALOG_PATH = DATA_PROCESSED / "catalog.csv"
TRANSACTIONS_PATH = DATA_PROCESSED / "transactions.csv.gz"
META_PATH = DATA_PROCESSED / "dataset_meta.json"

SEED = 42
CURRENCY = os.environ.get("FMCG_CURRENCY", "INR")

# Forecast horizon (weeks) of the lag-based forecaster. Lags are >= this value, so no look-ahead.
HORIZON_WEEKS = 4
# Reporting horizons (months -> weeks, 4.33 weeks/month rounded)
HORIZONS = {"1 month": 4, "3 months": 13, "6 months": 26, "12 months": 52}
COLD_START_WEEKS = 52          # weeks of product life modelled by the cold-start demand model

# Illustrative holiday calendar (ISO weeks). NOT an official festival calendar.
HOLIDAY_WEEKS = {10, 11, 42, 43, 44, 52}

REQUIRED_PANEL_COLS = ["date", "product_id", "store_id", "category", "region",
                       "store_type", "units", "price", "promo_flag"]
OPTIONAL_PANEL_COLS = ["subcategory", "brand", "pack_size", "pack_unit", "discount_pct",
                       "holiday_flag", "weeks_since_intro", "launch_observed", "positioning"]
REQUIRED_TRANSACTION_COLS = ["invoice_id", "customer_id", "invoice_date", "product_id",
                             "quantity", "unit_price"]

# ---- Documented commercial-hurdle rule (used instead of a trained success classifier) ----
HURDLE_RULE = (
    "A launch 'meets the commercial hurdle' when cumulative contribution over the evaluation "
    "window (default 26 weeks), AFTER deducting all fixed launch spend, is >= the user's target "
    "profit (default 0 = break-even). Units lost to stock-outs are not counted. "
    "Error calibration only uses products with enough observed weeks."
)

CHANNELS = ["Hypermarket", "Supermarket", "Convenience"]
POSITIONINGS = ["economy", "mass market", "premium", "super-premium"]
PACK_UNITS = ["g", "ml", "kg", "l"]

DISCLAIMER = (
    "Decision-support estimates only - not guarantees of commercial success. Values are labelled "
    "as Observed (historical data), Model estimate, or Assumption (user-entered)."
)
