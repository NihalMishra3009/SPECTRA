"""SQLite persistence for simulation runs (measurements + outcomes).

The database lives in the repo-level ``database/`` folder (override with the
``SPECTRA_DB_DIR`` environment variable). A run is written on every
``POST /api/simulate`` so the dashboard can list / replay / compare history.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_DIR = Path(os.environ.get("SPECTRA_DB_DIR", Path(__file__).resolve().parents[2] / "database"))
DB_FILE = Path(os.environ.get("SPECTRA_DB_FILE", DB_DIR / "spectra.db"))
SCHEMA_FILE = DB_DIR / "schema.sql"


# --------------------------------------------------------------------------- #
#  connection helpers
# --------------------------------------------------------------------------- #
def connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    if SCHEMA_FILE.exists():
        conn.executescript(SCHEMA_FILE.read_text())
        conn.commit()


def init() -> str:
    """Create the database + tables if needed. Returns the db path."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    conn.close()
    return str(DB_FILE)


# --------------------------------------------------------------------------- #
#  write
# --------------------------------------------------------------------------- #
def save_run(result: dict) -> str:
    """Persist a full simulation result. Returns the run_id."""
    run_id = str(uuid.uuid4())
    config = result.get("config", {})
    smart = result.get("smart", {})
    baseline = result.get("baseline", {})
    sm = smart.get("metrics", {})
    bm = baseline.get("metrics", {})

    outcome = {
        "smart_ir": sm.get("interception_ratio"),
        "baseline_ir": bm.get("interception_ratio"),
        "smart_reward": sm.get("avg_reward"),
        "baseline_reward": bm.get("avg_reward"),
        "miss_saved": max(0, bm.get("miss_count", 0) - sm.get("miss_count", 0)),
        "events": len(result.get("events", [])),
    }

    conn = connect()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO runs
                  (run_id, created_at, scenario_id, scenario_label, scheduler,
                   n_bands, n_steps, seed, baseline_metrics, smart_metrics,
                   outcome, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    result.get("scenario_id"),
                    result.get("scenario_label"),
                    config.get("scheduler") or smart.get("scheduler"),
                    config.get("n_bands"),
                    config.get("n_steps"),
                    config.get("seed"),
                    json.dumps(bm),
                    json.dumps(sm),
                    json.dumps(outcome),
                    json.dumps(result),
                ),
            )
        with conn:
            conn.executemany(
                """
                INSERT INTO telemetry (run_id, side, t, band, hit, snr, reward, ratio)
                VALUES (?, 'baseline', ?, ?, ?, ?, ?, ?)
                """,
                [
                    (run_id, e["t"], e["band"], int(e["hit"]), e.get("snr", 0.0), e["reward"], e["ratio"])
                    for e in baseline.get("log", [])
                ],
            )
            conn.executemany(
                """
                INSERT INTO telemetry (run_id, side, t, band, hit, snr, reward, ratio)
                VALUES (?, 'smart', ?, ?, ?, ?, ?, ?)
                """,
                [
                    (run_id, e["t"], e["band"], int(e["hit"]), e.get("snr", 0.0), e["reward"], e["ratio"])
                    for e in smart.get("log", [])
                ],
            )
            conn.executemany(
                """
                INSERT INTO events (run_id, t, type, bands_on, bands_off)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (run_id, e["t"], e["type"], json.dumps(e.get("bands_on", [])), json.dumps(e.get("bands_off", [])))
                    for e in result.get("events", [])
                ],
            )
        return run_id
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
#  read
# --------------------------------------------------------------------------- #
def list_runs(limit: int = 20) -> list[dict]:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT run_id, created_at, scenario_id, scenario_label, scheduler,
                   n_bands, n_steps, seed, outcome
            FROM runs ORDER BY id DESC LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) | {"outcome": json.loads(row["outcome"])} for row in rows]
    finally:
        conn.close()


def get_run(run_id: str) -> dict | None:
    conn = connect()
    try:
        row = conn.execute("SELECT result_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None
    finally:
        conn.close()


def stats() -> dict:
    conn = connect()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"]
        telemetry_rows = conn.execute("SELECT COUNT(*) c FROM telemetry").fetchone()["c"]
        scenario_breakdown = {
            r["scenario_id"]: r["c"]
            for r in conn.execute(
                "SELECT scenario_id, COUNT(*) c FROM runs GROUP BY scenario_id ORDER BY c DESC"
            )
        }
        best = conn.execute(
            """
            SELECT scenario_id, scheduler, json_extract(outcome, '$.smart_ir') ir
            FROM runs ORDER BY (json_extract(outcome, '$.smart_ir') IS NULL), ir DESC LIMIT 1
            """
        ).fetchone()
        return {
            "total_runs": total,
            "telemetry_samples": telemetry_rows,
            "scenario_breakdown": scenario_breakdown,
            "best_run": dict(best) if best else None,
            "db_path": str(DB_FILE),
        }
    except sqlite3.Error:
        return {"total_runs": 0, "telemetry_samples": 0, "scenario_breakdown": {}, "best_run": None, "db_path": str(DB_FILE)}
    finally:
        conn.close()


def last_run() -> dict | None:
    rows = list_runs(limit=1)
    return rows[0] if rows else None