"""Download the REAL datasets used by the Real Data Lab and record provenance (URL, SHA-256, rows).
Source: Rdatasets (github.com/vincentarelbundock/Rdatasets), which republishes datasets from the R packages
Ecdat, ISLR and tsibbledata (see each package's documentation for original citations and licences).
Files are already bundled in data/real/; run this only to refresh or re-verify them."""
import hashlib, json, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "data" / "real"
BASE = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
FILES = {"Cracker": "Ecdat", "Catsup": "Ecdat", "Ketchup": "Ecdat", "Tuna": "Ecdat", "Yogurt": "Ecdat", "OJ": "ISLR", "aus_retail": "tsibbledata"}
DESC = {"Cracker": "US supermarket scanner panel: cracker brand choice with prices, displays, features",
        "Catsup": "US supermarket scanner panel: ketchup brand/size choice with prices, displays, features",
        "Ketchup": "US supermarket scanner panel: ketchup brand choice with prices",
        "Tuna": "US supermarket scanner panel: canned tuna brand choice with prices",
        "Yogurt": "US supermarket scanner panel: yogurt brand choice with prices and features",
        "OJ": "Orange juice purchases (Citrus Hill vs Minute Maid) with store prices and specials",
        "aus_retail": "Australian Bureau of Statistics monthly retail turnover by state and industry (1982-2018)"}
def main(verify_only=False):
    OUT.mkdir(parents=True, exist_ok=True); prov = {}
    for name, pkg in FILES.items():
        p = OUT / f"{name}.csv"
        if not p.exists() or not verify_only:
            url = f"{BASE}/{pkg}/{name}.csv"; p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        else:
            url = f"{BASE}/{pkg}/{name}.csv"
        b = p.read_bytes()
        prov[name] = dict(package=pkg, url=url, sha256=hashlib.sha256(b).hexdigest(), rows=b.count(b"\n") - 1, description=DESC[name])
    (OUT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2)); print(json.dumps({k: v["rows"] for k, v in prov.items()}))
if __name__ == "__main__":
    main(verify_only="--verify" in sys.argv)
