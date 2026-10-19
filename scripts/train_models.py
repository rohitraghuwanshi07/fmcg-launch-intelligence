"""Train and save all models. Run after validate_data.py. Never invoked by the app at startup."""
import _bootstrap  # noqa
import json
import sys
from datetime import datetime, timezone
import numpy as np
from src import config, data_ingestion as di, forecasting as fc, pricing_analytics as pa, marketing_analytics as ma
from src import customer_segmentation as cseg, demo_data

def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o

def main() -> int:
    if not di.data_available():
        demo_data.generate_and_save()
    panel, meta = di.load_panel(), di.dataset_meta()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/5] Lag-based forecaster (chronological split) ...")
    lag = fc.train_lag_forecaster(panel)
    print("      selected:", lag["model_name"])
    print("[2/5] Cold-start launch-demand model (grouped CV on unseen products) ...")
    cs = fc.train_coldstart_model(panel)
    if cs is None:
        print("      SKIPPED: fewer than 20 products with an observed launch. Launch simulator unavailable.")
    else:
        print("      selected:", cs["model_name"], "| launches used:", cs["support"]["n_products"])
    print("[3/5] Price elasticity (observational) ...")
    el = pa.estimate_elasticity(panel)
    ela = {r.scope: float(r.elasticity) for r in el.itertuples() if not np.isnan(r.elasticity)}
    pooled = ela.pop("All categories (pooled)", -1.5)
    ela["pooled"] = pooled
    (config.MODELS_DIR / "pricing.json").write_text(json.dumps(dict(
        elasticity=ela, table=el.to_dict("records"), ratio_range=pa.supported_price_ratio_range(panel),
        promo=ma.promo_effect(panel), demo=meta.get("mode") != "real"), default=float, indent=2))
    print("[4/5] Customer segmentation ...")
    tx = di.load_transactions()
    seg_info = None
    if tx is not None:
        _, prof, seg_info = cseg.segment_customers(cseg.build_rfm(tx))
        print("      k =", seg_info["chosen_k"], "silhouette =", round(seg_info["silhouette"], 3))
    print("[5/5] Writing metadata ...")
    out = dict(trained_at=datetime.now(timezone.utc).isoformat(), dataset=meta, seed=config.SEED,
               lag_forecaster=dict(selected=lag["model_name"], train_end=lag["train_end"], val_end=lag["val_end"],
                                   metrics=lag["metrics"], interval_coverage_test=lag["interval_coverage_test"],
                                   wape_by_category=lag["wape_by_category"], n=[lag["n_train"], lag["n_val"], lag["n_test"]]),
               coldstart=None if cs is None else dict(selected=cs["model_name"], metrics=cs["metrics"], importance=cs["importance"],
                                                      support=cs["support"], log_ratio_sd=float(np.std(cs["log_ratio"])),
                                                      weekly_noise_sd=cs["weekly_noise_sd"], n_launches=cs["n_ratio_products"]),
               segmentation=seg_info)
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(_jsonable(out), indent=2, default=float))
    print("Done. Metrics in models/metrics.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
