"""SQLite persistence for saved launch scenarios."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from . import config

def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS scenarios (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                 created_at TEXT NOT NULL, dataset_mode TEXT, inputs_json TEXT NOT NULL, summary_json TEXT NOT NULL)""")
    return c

def save_scenario(name: str, inputs: dict, summary: dict, dataset_mode: str = "demo_synthetic") -> int:
    if not name.strip():
        raise ValueError("Scenario name cannot be empty.")
    with _conn() as c:
        cur = c.execute("INSERT INTO scenarios (name, created_at, dataset_mode, inputs_json, summary_json) VALUES (?,?,?,?,?)",
                        (name.strip(), datetime.now(timezone.utc).isoformat(timespec="seconds"), dataset_mode,
                         json.dumps(inputs, default=str), json.dumps(summary, default=float)))
        return cur.lastrowid

def list_scenarios() -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT id, name, created_at, dataset_mode, inputs_json, summary_json FROM scenarios ORDER BY id DESC").fetchall()
    return [dict(id=r[0], name=r[1], created_at=r[2], dataset_mode=r[3], inputs=json.loads(r[4]), summary=json.loads(r[5])) for r in rows]

def get_scenario(sid: int) -> dict | None:
    return next((s for s in list_scenarios() if s["id"] == sid), None)

def delete_scenario(sid: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM scenarios WHERE id=?", (sid,))
