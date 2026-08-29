"""Tests for the SQLite persistence layer + the /api/db/* endpoints."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import SimConfig
from app.main import app
from app.sim.engine import run_simulation

SCHEMA = Path(db.__file__).resolve().parents[2] / "database" / "schema.sql"


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Redirect the database into a temp dir for the duration of the test."""
    db_file = tmp_path / "spectra.db"
    monkeypatch.setattr(db, "DB_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_FILE", db_file)
    monkeypatch.setattr(db, "SCHEMA_FILE", SCHEMA)
    # WAL sidecar files stay inside tmp too — nothing leaks into the repo.
    return db_file


def _sample_run():
    return run_simulation(
        SimConfig(n_bands=8, n_steps=40, seed=7, scenario="switch", scheduler="thompson")
    )


def test_init_creates_database_and_tables(fresh_db):
    path = db.init()
    assert Path(path).exists()
    conn = db.connect()
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"runs", "telemetry", "events"} <= tables


def test_save_and_list_run(fresh_db):
    rid = db.save_run(_sample_run())
    runs = db.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == rid
    assert runs[0]["scenario_id"] == "switch"
    assert runs[0]["scheduler"] == "thompson"
    assert runs[0]["outcome"]["smart_ir"] is not None


def test_get_run_roundtrip(fresh_db):
    rid = db.save_run(_sample_run())
    got = db.get_run(rid)
    assert got is not None
    assert got["scenario_id"] == "switch"
    assert got["smart"]["metrics"]["interception_ratio"] is not None
    assert db.get_run("does-not-exist") is None


def test_telemetry_and_events_persisted(fresh_db):
    result = _sample_run()
    rid = db.save_run(result)
    conn = db.connect()
    n_telem = conn.execute("SELECT COUNT(*) c FROM telemetry WHERE run_id=?", (rid,)).fetchone()["c"]
    n_events = conn.execute("SELECT COUNT(*) c FROM events WHERE run_id=?", (rid,)).fetchone()["c"]
    conn.close()
    assert n_telem == 2 * len(result["smart"]["log"])
    assert n_events == len(result["events"])


def test_stats(fresh_db):
    a = db.save_run(_sample_run())
    b = db.save_run(_sample_run())
    s = db.stats()
    assert s["total_runs"] == 2
    assert s["telemetry_samples"] == 2 * 2 * 40
    assert s["scenario_breakdown"]["switch"] == 2
    assert s["best_run"] is not None


def test_api_simulate_records_run_and_endpoints(fresh_db):
    client = TestClient(app)
    r = client.post(
        "/api/simulate", json={"scenario": "stable", "scheduler": "ucb1", "n_steps": 30, "seed": 3, "n_bands": 10}
    )
    assert r.status_code == 200
    rid = r.json()["run_id"]
    assert rid

    runs = client.get("/api/db/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["run_id"] == rid

    detail = client.get(f"/api/db/runs/{rid}").json()
    assert detail["scenario_id"] == "stable"
    assert detail["smart"]["metrics"]["hits"] >= 0

    stats = client.get("/api/db/stats").json()
    assert stats["total_runs"] == 1
    assert stats["telemetry_samples"] == 2 * 30

    assert client.get("/api/db/runs/nope").status_code == 404