import numpy as np
import pandas as pd
import pytest
from src import data_ingestion as di, data_validation as dv, config


def _panel(n=60):
    d = pd.date_range("2024-01-01", periods=n, freq="W-MON")
    return pd.DataFrame(dict(date=d, product_id="P1", store_id="S1", category="Home Care", region="North", store_type="Supermarket",
                             units=np.arange(n) % 7 + 5, price=10.0, promo_flag=0))

def test_valid_panel_passes():
    assert dv.validate_panel(_panel())["errors"] == []

def test_missing_columns_and_empty():
    assert "Missing required columns" in dv.validate_panel(_panel().drop(columns=["price"]))["errors"][0]
    assert dv.validate_panel(pd.DataFrame())["errors"] == ["Dataset is empty."]

def test_negative_price_units_and_duplicates_detected():
    p = _panel(); p.loc[0, "price"] = -1; p.loc[1, "units"] = -5; p = pd.concat([p, p.iloc[[2]]])
    msgs = " ".join(dv.validate_panel(p)["errors"])
    assert "negative units" in msgs and "non-positive price" in msgs and "duplicate" in msgs

def test_quality_report_handles_bool_and_missing():
    p = _panel(); p["flag"] = True; p.loc[3, "units"] = np.nan
    q = dv.quality_report(p).set_index("column")
    assert q.loc["units", "missing"] == 1 and q.loc["flag", "dtype"] == "bool"

def test_competitor_upload_validation():
    ok = pd.DataFrame([dict(product_name="A", brand="B", category="Home Care", pack_size=100, pack_unit="g", price=30, currency=config.CURRENCY,
                            price_type="regular", source_name="audit", source_url="n/a", retrieved_at="2025-01-01", market="Delhi")])
    assert dv.validate_competitor_upload(ok)["errors"] == []
    bad = ok.assign(currency="USD", pack_unit="oz", price_type="weird", price=-1)
    assert len(dv.validate_competitor_upload(bad)["errors"]) >= 4
    assert dv.validate_competitor_upload(ok.drop(columns=["source_url"]))["errors"]

def test_launch_input_validation_messages():
    from src.scenario_engine import LaunchInputs
    i = LaunchInputs(price=-5, ad_budget=-1, intro_discount=0.8, regions=[])
    msgs = " ".join(i.validate()["errors"])
    assert "'price'" in msgs and "'ad_budget'" in msgs and "discount" in msgs.lower() and "region" in msgs.lower()
    assert LaunchInputs(total_launch_budget=100).validate()["errors"]          # budget lines exceed total
    assert LaunchInputs(launch_date="not-a-date").validate()["errors"]
    assert LaunchInputs(currency="USD").validate()["errors"]
    assert LaunchInputs(mfg_cost=40).validate()["warnings"]                    # loses money per unit

def test_online_retail_adapter_on_format_fixture():
    """Format fixture mimicking the Online Retail II schema (NOT real data)."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2010-01-04", periods=400, freq="D")
    rows = []
    for i, d in enumerate(days):
        for code, desc in (("A1", "RED MUG"), ("B2", "CHRISTMAS CARD"), ("C3", "WOODEN FRAME")):
            if code == "C3" and i < 120: continue
            rows.append((f"5{i}{code}", code, desc, int(rng.integers(1, 10)), d, 2.5, 12345 + int(rng.integers(0, 50)), "United Kingdom"))
    rows.append(("C99", "A1", "RED MUG", -3, days[5], 2.5, 1, "United Kingdom"))          # cancellation
    rows.append(("5999", "POST", "POSTAGE", 1, days[5], 20, 1, "United Kingdom"))        # non-product
    raw = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
    panel, cat, tx = di.adapt_online_retail_ii(raw)
    assert not tx.invoice_id.str.startswith("C").any() and "POST" not in set(panel.product_id)
    assert set(cat.category) <= {"Kitchen & Tableware", "Seasonal & Gifting", "Home Decor", "Other"}
    gaps = panel.sort_values("date").groupby(["product_id", "store_id"]).date.diff().dropna()
    assert (gaps == pd.Timedelta(days=7)).all()
    assert cat.set_index("product_id").loc["C3", "launch_observed"]                    # started late -> launch observed
    assert dv.validate_panel(panel)["errors"] == []

def test_unit_conversion():
    assert di.to_base_size(1, "kg") == (1000, "g") and di.to_base_size(2, "l") == (2000, "ml")
