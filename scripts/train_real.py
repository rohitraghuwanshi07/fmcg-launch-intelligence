"""Estimate everything that uses REAL data: scanner-panel elasticities + real retail forecasting benchmark (~2-3 min)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
from src import real_data as rd
out = rd.build_real_artifacts()
print({c: round(v["elasticity"], 2) for c, v in out["elasticities"]["categories"].items()})
for r in out["benchmark"]["by_horizon"]:
    print(f"h={r['h']:>2}  GBM WAPE {r['GBM (global)|WAPE']:.3f}  seasonal-naive {r['Seasonal naive|WAPE']:.3f}")
print("stat benchmarks:", [(x['model'], round(x['mean'], 3)) for x in out["benchmark"]["stat_benchmarks"]])
