# SPECTRA · database/

SQLite persistence for the simulated RF environment — the "measurements" data
product the scheduler is trained against (PS 26055).

## Layout

| File | Purpose |
|---|---|
| `schema.sql` | Tables: `runs`, `telemetry` (per-step per-receiver measurements), `events` |
| `spectra.db` | Runtime database file — **generated automatically, not committed** |

## How it works

- **Automatic**: every `POST /api/simulate` writes the full deterministic run
  (metrics + outcome + per-step telemetry + ground-truth events) into
  `database/spectra.db`. The response gains a `run_id`.
- **Override**: set `SPECTRA_DB_DIR` / `SPECTRA_DB_FILE` to move it (tests do
  this to a temp dir).
- **README of the backend**: `backend/app/db.py` is the only wrapper.

## API

| Endpoint | What |
|---|---|
| `GET /api/db/runs` | recent runs (scenario, scheduler, outcome, Δ) |
| `GET /api/db/runs/{run_id}` | full stored run — dashboard replays it |
| `GET /api/db/stats` | totals, telemetry sample count, scenario breakdown, best run |

## Populate sample history

```powershell
cd backend
.\.venv\Scripts\python -m app.seed_db        # 6 varied runs across schedulers
```

Seed script trains no models — just records representative runs so the
dashboard's "Recorded runs" panel and stats are populated before your first
manual click.