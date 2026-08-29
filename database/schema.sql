-- SPECTRA — database schema (SQLite)
-- Stores measurements & outcomes from the simulated RF environment (PS 26055).

PRAGMA journal_mode = WAL;

-- One line per complete simulation run
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL UNIQUE,          -- public identifier (uuid)
    created_at        TEXT    NOT NULL,                 -- ISO-8601
    scenario_id       TEXT    NOT NULL,
    scenario_label    TEXT,
    scheduler         TEXT    NOT NULL,
    n_bands           INTEGER NOT NULL,
    n_steps           INTEGER NOT NULL,
    seed              INTEGER,
    baseline_metrics  TEXT,                             -- JSON (7 PS figures of merit)
    smart_metrics     TEXT,                             -- JSON (7 PS figures of merit)
    outcome           TEXT    NOT NULL,                 -- JSON summary blob (ir, reward, deltas)
    result_json       TEXT,                             -- full deterministic run (replay via API)
    CHECK (n_bands > 0 AND n_steps > 0)
);

-- Per-step RF measurements: what each receiver observed at every time step
CREATE TABLE IF NOT EXISTS telemetry (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    side      TEXT    NOT NULL CHECK (side IN ('baseline', 'smart')),
    t         INTEGER NOT NULL,
    band      INTEGER NOT NULL,
    hit       INTEGER NOT NULL CHECK (hit IN (0, 1)),
    snr       REAL    NOT NULL DEFAULT 0.0,
    reward    REAL    NOT NULL DEFAULT 0.0,
    ratio     REAL    NOT NULL DEFAULT 0.0,             -- cumulative interception ratio at t
    UNIQUE (run_id, side, t)
);

-- Ground-truth emitter events (pattern switch / surprise) surfaced during a run
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT    NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    t         INTEGER NOT NULL,
    type      TEXT    NOT NULL CHECK (type IN ('change', 'surprise')),
    bands_on  TEXT,                                    -- JSON array
    bands_off TEXT                                     -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_telemetry_run ON telemetry(run_id, side);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at DESC);