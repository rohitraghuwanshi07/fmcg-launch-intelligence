"""Test isolation: all data/models/DB go to a temp dir BEFORE any src module is imported."""
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fmcg_tests_"))
os.environ["FMCG_PROCESSED_DIR"] = str(_TMP / "processed")
os.environ["FMCG_MODELS_DIR"] = str(_TMP / "models")
os.environ["FMCG_DB_PATH"] = str(_TMP / "scenarios.db")
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{_TMP / 'app.db'}"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["AUTH_RATE_LIMIT"] = "1000"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    """Generate a small SIMULATED dataset and run the real training script on it."""
    from src import demo_data, config
    demo_data.generate_and_save(n_products=36, n_customers=400)
    import train_models
    assert train_models.main() == 0
    return config


@pytest.fixture(scope="session")
def ctx(trained):
    from src import scenario_engine as se
    c = se.load_context()
    assert c is not None
    return c


@pytest.fixture(scope="session")
def panel(trained):
    from src import data_ingestion as di
    return di.load_panel()
