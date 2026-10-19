"""End-to-end API tests: auth, ownership isolation, persistence, migrations, jobs, analytics endpoints."""
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained):
    from backend.app.main import app
    with TestClient(app) as c:          # runs lifespan: alembic upgrade + model warm-up
        yield c


def _signup(c, email, name="Tester"):
    r = c.post("/api/auth/register", json={"email": email, "name": name, "password": "correct-horse-battery"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def alice(client): return _signup(client, "alice@example.com", "Alice")


@pytest.fixture(scope="module")
def bob(client): return _signup(client, "bob@example.com", "Bob")


@pytest.fixture(scope="module")
def project(client, alice):
    r = client.post("/api/projects", json={"name": "Crunchy Bites"}, headers=alice)
    assert r.status_code == 201; return r.json()


def _wait(c, h, jid, timeout=150):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}", headers=h).json()
        if j["status"] in ("done", "failed"): return j
        time.sleep(0.5)
    raise AssertionError("job timed out")


def test_health_and_meta_are_public(client):
    assert client.get("/api/health").json()["models_loaded"]
    m = client.get("/api/meta").json()
    assert m["ready"] and m["categories"] and m["defaults"]["price"] > 0


def test_auth_flow_and_error_cases(client):
    h = _signup(client, "carol@example.com")
    assert client.get("/api/auth/me", headers=h).json()["email"] == "carol@example.com"
    assert client.post("/api/auth/register", json={"email": "CAROL@example.com", "name": "x", "password": "correct-horse-battery"}).status_code == 409
    assert client.post("/api/auth/register", json={"email": "bad", "name": "x", "password": "correct-horse-battery"}).status_code == 422
    assert client.post("/api/auth/register", json={"email": "d@example.com", "name": "x", "password": "short"}).status_code == 422
    assert client.post("/api/auth/login", json={"email": "carol@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong-password"}).json()["detail"] == \
        client.post("/api/auth/login", json={"email": "carol@example.com", "password": "wrong-password"}).json()["detail"]   # no user enumeration
    ok = client.post("/api/auth/login", json={"email": "carol@example.com", "password": "correct-horse-battery"})
    assert ok.status_code == 200 and ok.json()["access_token"]


def test_protected_routes_reject_missing_or_forged_tokens(client):
    for path in ("/api/projects", "/api/auth/me", "/api/analytics/overview", "/api/real/benchmark"):
        assert client.get(path).status_code == 401
    assert client.get("/api/projects", headers={"Authorization": "Bearer not.a.token"}).status_code == 401


def test_passwords_are_hashed(client, trained):
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import User
    with SessionLocal() as db:
        u = db.query(User).first()
        assert u.password_hash.startswith("$2") and "correct-horse" not in u.password_hash


def test_project_crud_and_ownership_isolation(client, alice, bob, project):
    pid = project["id"]
    assert [p["id"] for p in client.get("/api/projects", headers=alice).json()] == [pid]
    assert client.get("/api/projects", headers=bob).json() == []
    for method, path in (("get", f"/api/projects/{pid}"), ("delete", f"/api/projects/{pid}"), ("post", f"/api/projects/{pid}/assess"),
                         ("get", f"/api/projects/{pid}/latest"), ("get", f"/api/projects/{pid}/report.pdf")):
        assert getattr(client, method)(path, headers=bob).status_code == 404, path       # 404, never 403
    r = client.put(f"/api/projects/{pid}", json={"name": "Renamed"}, headers=alice)
    assert r.status_code == 200 and r.json()["name"] == "Renamed"


def test_assessment_is_persisted_versioned_and_goes_stale_after_edit(client, alice, project):
    pid = project["id"]
    assert client.get(f"/api/projects/{pid}/latest", headers=alice).status_code == 404
    r = client.post(f"/api/projects/{pid}/assess", json={"n_sims": 400}, headers=alice)
    assert r.status_code == 200, r.text
    a = r.json()
    assert 0 <= a["p_hurdle"] <= 1 and a["readiness"]["grade"] in "ABCD" and a["decision"]["decision"] and (a["shap"] is None or a["shap"]["table"]) and a["conformal"]["low"] < a["conformal"]["high"]
    assert not a["stale"] and client.get(f"/api/projects/{pid}/latest", headers=alice).json()["assessment_id"] == a["assessment_id"]
    inputs = client.get(f"/api/projects/{pid}", headers=alice).json()["inputs"]; inputs["price"] = inputs["price"] * 1.1
    client.put(f"/api/projects/{pid}", json={"inputs": inputs}, headers=alice)
    assert client.get(f"/api/projects/{pid}/latest", headers=alice).json()["stale"]          # plan changed since that assessment
    client.post(f"/api/projects/{pid}/assess", json={"n_sims": 400}, headers=alice)
    assert len(client.get(f"/api/projects/{pid}/assessments", headers=alice).json()) == 2


def test_invalid_plan_is_rejected_with_clear_message(client, alice):
    p = client.post("/api/projects", json={"name": "Bad"}, headers=alice).json()
    inputs = p["inputs"]; inputs["price"] = -5
    client.put(f"/api/projects/{p['id']}", json={"inputs": inputs}, headers=alice)
    r = client.post(f"/api/projects/{p['id']}/assess", json={"n_sims": 400}, headers=alice)
    assert r.status_code == 422 and "price" in r.json()["detail"]


def test_sync_analysis_endpoints(client, alice, project):
    pid = project["id"]
    w = client.post(f"/api/projects/{pid}/whatif", json={"price_pct": -10}, headers=alice).json()
    assert w["scenario"]["units_6m"] > w["base"]["units_6m"]                                   # lower price -> more units
    assert client.post(f"/api/projects/{pid}/whatif", json={"price_pct": 999}, headers=alice).status_code == 422
    an = client.get(f"/api/projects/{pid}/analogs?k=5", headers=alice).json()
    assert an["available"] and an["ensemble"]["agreement"] in ("Strong", "Moderate", "Weak")
    s = client.get(f"/api/projects/{pid}/sensitivity", headers=alice).json()
    assert s["whatifs"] and s["tornado"] and s["price_curve"]
    q = client.post(f"/api/projects/{pid}/ask", json={"question": "what is the downside risk?"}, headers=alice).json()
    assert q["answer"]["check"]["grounded"] and q["brief"]
    pdf = client.get(f"/api/projects/{pid}/report.pdf", headers=alice)
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_jobs_run_in_background_persist_and_are_private(client, alice, bob, project):
    pid = project["id"]
    r = client.post(f"/api/projects/{pid}/jobs", json={"kind": "pilot", "params": {"rounds": 6}}, headers=alice)
    assert r.status_code == 202
    jid = r.json()["id"]
    assert client.get(f"/api/jobs/{jid}", headers=bob).status_code == 404
    j = _wait(client, alice, jid)
    assert j["status"] == "done" and set(j["result"]["policies"]) == {"Uniform", "Epsilon-greedy", "Thompson sampling"}
    assert client.post(f"/api/projects/{pid}/jobs", json={"kind": "rm -rf"}, headers=alice).status_code == 422
    assert any(x["id"] == jid for x in client.get(f"/api/projects/{pid}/jobs?kind=pilot", headers=alice).json())


def test_red_team_job_result_is_consistent(client, alice, project):
    jid = client.post(f"/api/projects/{project['id']}/jobs", json={"kind": "red_team", "params": {"threshold": 0.5}}, headers=alice).json()["id"]
    j = _wait(client, alice, jid)
    assert j["status"] == "done", j["error"]
    R = j["result"]
    assert R["found"] and R["p_at_failure"] <= 0.52 and "missed its hurdle" in R["pre_mortem"] and abs(sum(R["shares"].values()) - 1) < 1e-9


def test_failed_job_reports_error_without_traceback(client, alice, project):
    from backend.app.services import jobs
    jid = jobs.submit(1, project["id"], "pilot", {"category": "NOPE"}, {})         # corrupt inputs -> task raises
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import Job
    for _ in range(60):
        with SessionLocal() as db:
            j = db.get(Job, jid)
            if j.status in ("failed", "done"): break
        time.sleep(0.2)
    assert j.status == "failed" and j.error and "Traceback" not in j.error


def test_analytics_models_and_real_data_endpoints(client, alice):
    ov = client.get("/api/analytics/overview", headers=alice).json()
    assert ov["kpis"]["revenue"] > 0 and ov["monthly"] and ov["categories"]
    assert client.get("/api/models/metrics", headers=alice).status_code == 200
    el = client.get("/api/real/elasticities", headers=alice).json()
    assert el["datasets"]["Cracker"]["elasticity"] < 0 and "Biscuits & Snacks" in el["categories"]
    assert client.get("/api/real/benchmark", headers=alice).json()["by_horizon"]
    s = client.get("/api/real/series", headers=alice).json()
    f = client.get(f"/api/real/series/{s[0]['sid']}", headers=alice).json()
    assert len(f["forecast"]) == 12 and f["history"]
    assert client.get("/api/real/seasonality?industry=Nope", headers=alice).status_code == 404
    assert client.get("/api/about").json()["test_functions"] > 40


def test_schema_matches_migrations(client):
    from sqlalchemy import inspect
    from backend.app.db.database import engine, Base
    insp = inspect(engine)
    assert "alembic_version" in insp.get_table_names()
    for t in Base.metadata.tables.values():
        assert {c["name"] for c in insp.get_columns(t.name)} == {c.name for c in t.columns}, t.name
