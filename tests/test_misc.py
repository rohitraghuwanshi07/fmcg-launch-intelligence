import json
import pandas as pd
import pytest
from src import storage, reporting as rp, customer_segmentation as cseg, success_model as sm, data_ingestion as di
from src import scenario_engine as se, recommendation_engine as rec, market_intelligence as mi


def test_storage_roundtrip():
    sid = storage.save_scenario("  My plan ", {"price": 10}, {"p": 0.5}, "demo_synthetic")
    s = storage.get_scenario(sid)
    assert s["name"] == "My plan" and s["inputs"]["price"] == 10 and s["summary"]["p"] == 0.5
    storage.delete_scenario(sid); assert storage.get_scenario(sid) is None
    with pytest.raises(ValueError): storage.save_scenario("  ", {}, {})

def test_segmentation(trained):
    tx = di.load_transactions()
    rfm = cseg.build_rfm(tx)
    seg, prof, info = cseg.segment_customers(rfm)
    assert len(seg) == len(rfm) and -1 <= info["silhouette"] <= 1 and prof.customers.sum() == len(rfm)
    with pytest.raises(ValueError): cseg.segment_customers(rfm.head(10))
    with pytest.raises(ValueError): cseg.build_rfm(tx.iloc[0:0])
    assert not cseg.basket_pairs(tx, min_support=0.0001).empty

def test_classifier_refuses_without_labels():
    assert sm.classifier_status(None)["available"] is False
    assert sm.classifier_status(pd.DataFrame({"x": [1]}))["available"] is False

def test_market_intelligence(panel, trained):
    ref = mi.catalog_reference(panel, di.load_catalog())
    assert "HISTORICAL" in ref.price_basis.iloc[0] and (ref.ppu100 > 0).all()
    assert mi.normalise_ppu100(30, 100, "g") == 30 and mi.normalise_ppu100(30, 1, "kg") == 3
    assert mi.freshness_status("2000-01-01").startswith("STALE") and mi.freshness_status("garbage") == "unknown"
    view = ref[ref.category == "Home Care"]
    pos = mi.positioning_summary(view.ppu100.median(), view)
    assert 30 < pos["percentile"] < 70 and not mi.positioning_summary(1, view.iloc[0:0])["available"]

def test_end_to_end_documented_sample_input(ctx):
    """Documented sample: a 100 g biscuit pack launched in North+South (see README)."""
    sample = dict(product_name="Demo Crunch", category="Biscuits & Snacks", pack_size=100, pack_unit="g", price=35, mfg_cost=10, packaging_cost=2,
                  dist_cost=1.5, regions=["North", "South"], launch_date="2025-03-03", outlets={"Hypermarket": 20, "Supermarket": 80, "Convenience": 150})
    inp = se.LaunchInputs.from_dict(sample); assert inp.validate()["errors"] == []
    res = se.run_assessment(inp, ctx)
    recs, dec = rec.generate(res, ctx), rec.decide(res)
    assert dec["decision"] and 0 <= res["p_hurdle"] <= 1
    pdf1, pdf2 = rp.executive_pdf(res, recs, dec, di.dataset_meta()), rp.detailed_pdf(res, recs, dec, di.dataset_meta(), None, json.loads((__import__("src").config.MODELS_DIR / "metrics.json").read_text()))
    assert pdf1[:5] == b"%PDF-" and pdf2[:5] == b"%PDF-" and len(pdf2) > len(pdf1) > 2000
    csvs = rp.results_csvs(res, recs)
    assert {"horizons.csv", "scenarios.csv", "weekly_forecast.csv", "recommendations.csv"} <= set(csvs)
