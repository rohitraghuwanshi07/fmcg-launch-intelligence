"""End-to-end smoke test against a REAL server process (uvicorn + SQLite + built frontend).
Usage: python scripts/smoke_e2e.py        (exits non-zero on the first failed check)"""
import json, os, subprocess, sys, tempfile, time, urllib.error, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; PORT = int(os.environ.get("SMOKE_PORT", "8765")); BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, body=None, token=None, raw=False):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        r = urllib.request.urlopen(req, timeout=120); data = r.read()
        return r.status, (data if raw else json.loads(data or b"null")), r.headers
    except urllib.error.HTTPError as e:
        data = e.read()
        try: return e.code, json.loads(data), e.headers
        except Exception: return e.code, data, e.headers


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok: raise SystemExit(1)


def main():
    tmp = tempfile.mkdtemp()
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp}/smoke.db", "SECRET_KEY": "smoke-secret", "AUTH_RATE_LIMIT": "500"}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(PORT)], cwd=ROOT, env=env,
                           stdout=open(f"{tmp}/server.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(90):
            try:
                if urllib.request.urlopen(BASE + "/api/health", timeout=2).status == 200: break
            except Exception: time.sleep(1)
        else:
            print(open(f"{tmp}/server.log").read()[-2000:]); raise SystemExit("server did not start")
        s, h, _ = call("GET", "/api/health"); check("health", s == 200 and h["models_loaded"], str(h))
        if (ROOT / "frontend" / "dist" / "index.html").exists():
            s, body, hd = call("GET", "/", raw=True); check("serves SPA index", s == 200 and b"<div id=\"root\">" in body)
            s, body, _ = call("GET", "/projects/1/assessment", raw=True); check("SPA deep-link fallback", s == 200 and b"<div id=\"root\">" in body)
        s, d, _ = call("GET", "/api/nope"); check("unknown API route is a JSON 404", s == 404)
        s, r, _ = call("POST", "/api/auth/register", {"email": "smoke@example.com", "name": "Smoke", "password": "correct-horse-battery"}); check("register", s == 201)
        tok = r["access_token"]
        s, _, _ = call("GET", "/api/projects"); check("projects require auth", s == 401)
        s, p, _ = call("POST", "/api/projects", {"name": "Smoke plan"}, tok); check("create project", s == 201); pid = p["id"]
        t0 = time.time(); s, a, _ = call("POST", f"/api/projects/{pid}/assess", {"n_sims": 1000}, tok)
        check("assess", s == 200 and 0 <= a["p_hurdle"] <= 1, f"{a['decision']['decision']}, P={a['p_hurdle']:.2f}, grade {a['readiness']['grade']}, {time.time() - t0:.1f}s")
        check("real price response in use", a["price_response"]["source"] in ("real scanner data", "simulated panel"), str(a["price_response"]))
        s, w, _ = call("POST", f"/api/projects/{pid}/whatif", {"price_pct": -10}, tok); check("what-if", s == 200 and w["scenario"]["units_6m"] > w["base"]["units_6m"])
        for kind, params in (("red_team", {"threshold": 0.5}), ("pilot", {"rounds": 6}), ("optimize", {"lam": 0.5})):
            s, j, _ = call("POST", f"/api/projects/{pid}/jobs", {"kind": kind, "params": params}, tok); check(f"{kind}: job accepted", s == 202)
            t0 = time.time()
            while time.time() - t0 < 180:
                s, jj, _ = call("GET", f"/api/jobs/{j['id']}", token=tok)
                if jj["status"] in ("done", "failed"): break
                time.sleep(1)
            check(f"{kind}: job done", jj["status"] == "done", f"{time.time() - t0:.0f}s {jj.get('error') or ''}")
            if kind == "red_team": print("     pre-mortem:", jj["result"]["pre_mortem"][:150])
        s, pdf, _ = call("GET", f"/api/projects/{pid}/report.pdf", token=tok, raw=True); check("pdf report", s == 200 and pdf[:4] == b"%PDF", f"{len(pdf)} bytes")
        s, el, _ = call("GET", "/api/real/elasticities", token=tok); check("real-data elasticities", s == 200 and el["datasets"]["Cracker"]["elasticity"] < 0)
        s, o, _ = call("GET", "/api/analytics/overview", token=tok); check("portfolio overview", s == 200 and o["kpis"]["revenue"] > 0)
        s, r2, _ = call("POST", "/api/auth/register", {"email": "other@example.com", "name": "Other", "password": "correct-horse-battery"})
        s, _, _ = call("GET", f"/api/projects/{pid}", token=r2["access_token"]); check("another user cannot read the project (404)", s == 404)
        s, m, hd = call("GET", "/api/health"); check("request-id header", bool(hd.get("X-Request-ID")))
        print("\nALL SMOKE CHECKS PASSED")
    finally:
        srv.terminate()
        try: srv.wait(10)
        except Exception: srv.kill()


if __name__ == "__main__":
    main()
