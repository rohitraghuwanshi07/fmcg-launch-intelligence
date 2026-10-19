"""Download the real Online Retail II dataset (UCI) and import it through the adapter.

Requires internet access to archive.ics.uci.edu. If the download fails, download manually from
https://archive.ics.uci.edu/dataset/502/online+retail+ii and place the .xlsx/.csv in data/raw/.
No credentials are required. Check the dataset licence (CC BY 4.0 per UCI at time of writing) before use.
NOTE: written and unit-tested against a format fixture only; the live download could not be tested
in the build sandbox (network egress blocked).
"""
import _bootstrap  # noqa
import io
import sys
import urllib.request
import zipfile
import pandas as pd
from src import config, data_ingestion as di

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

def main() -> int:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    local = next(iter(list(config.DATA_RAW.glob("online_retail_II*")) + list(config.DATA_RAW.glob("online*retail*"))), None)
    if local is None:
        print("Downloading", URL)
        try:
            with urllib.request.urlopen(URL, timeout=60) as r:
                data = r.read()
        except Exception as e:
            print(f"Download failed ({e}). Download manually and place the file in {config.DATA_RAW}.")
            return 1
        z = zipfile.ZipFile(io.BytesIO(data)); z.extractall(config.DATA_RAW)
        local = next(config.DATA_RAW.glob("*.xlsx"), None) or next(config.DATA_RAW.glob("*.csv"))
    di.record_provenance("Online Retail II", URL, local, "CC BY 4.0 (verify at source)",
                         "UK gift wholesaler transactions 2009-2011; not an FMCG manufacturer dataset")
    print("Reading", local, "(Excel files can take minutes) ...")
    raw = pd.concat(pd.read_excel(local, sheet_name=None).values()) if local.suffix == ".xlsx" else pd.read_csv(local)
    panel, cat, tx = di.adapt_online_retail_ii(raw)
    meta = di.import_real_dataset(panel, cat, tx, "Online Retail II (UCI)", URL, "CC BY 4.0 (verify)", "GBP")
    print(meta)
    print("Now run: python scripts/validate_data.py && python scripts/train_models.py")
    print("Caveat: prices are GBP and the app's launch form is INR-based; set FMCG_CURRENCY=GBP.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
