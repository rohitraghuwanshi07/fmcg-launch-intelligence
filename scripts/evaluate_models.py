"""Print the saved evaluation report (no retraining)."""
import _bootstrap  # noqa
import json
from src import config

def main():
    p = config.MODELS_DIR / "metrics.json"
    if not p.exists():
        print("No metrics found. Run scripts/train_models.py first."); return 1
    m = json.loads(p.read_text())
    print(f"Dataset: {m['dataset'].get('name')} (mode={m['dataset'].get('mode')})")
    print("\n== Lag forecaster: TEST (chronological hold-out) ==")
    for k, v in m["lag_forecaster"]["metrics"]["test"].items():
        print(f"{k:42s} WAPE={v['WAPE']:.3f} MAE={v['MAE']:.2f} RMSE={v['RMSE']:.2f} MASE={v.get('MASE', float('nan')):.2f}")
    print("80% interval coverage on test:", round(m["lag_forecaster"]["interval_coverage_test"], 3))
    if m.get("coldstart"):
        print("\n== Cold-start launch model: grouped CV (dev products) ==")
        for k, v in m["coldstart"]["metrics"]["cv_dev"].items():
            print(f"{k:34s} WAPE={v['WAPE']:.3f} MAE={v['MAE']:.2f} product MdAPE(26w)={v['product_MdAPE_26w']:.3f}")
        print("== held-out products (selected on CV, evaluated once) ==")
        for k, v in m["coldstart"]["metrics"]["test"].items():
            print(f"{k:44s} WAPE={v['WAPE']:.3f} MAE={v['MAE']:.2f} product MdAPE(26w)={v['product_MdAPE_26w']:.3f} (n={v['n_products']})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
