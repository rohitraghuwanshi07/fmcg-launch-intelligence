"""Validate the active dataset; create the labelled demo dataset if none exists."""
import _bootstrap  # noqa
import sys
from src import config, data_ingestion as di, data_validation as dv, demo_data

def main() -> int:
    if not di.data_available():
        print("No dataset found -> generating the SIMULATED demo dataset (clearly labelled).")
        demo_data.generate_and_save()
    meta = di.dataset_meta()
    print(f"Dataset: {meta.get('name')} | mode={meta.get('mode')} | {meta.get('start')} -> {meta.get('end')}")
    panel = di.load_panel()
    res = dv.validate_panel(panel)
    for w in res["warnings"]: print("WARNING:", w)
    for e in res["errors"]: print("ERROR:", e)
    print(f"Panel rows: {len(panel):,} | products: {panel.product_id.nunique()} | stores: {panel.store_id.nunique()}")
    print("Launch-observed products:", int(panel.loc[panel.launch_observed.astype(bool), 'product_id'].nunique()) if 'launch_observed' in panel else 0)
    if meta.get("mode") != "real":
        print("NOTE: demo/synthetic data - model metrics are demonstration only.")
    return 1 if res["errors"] else 0

if __name__ == "__main__":
    sys.exit(main())
