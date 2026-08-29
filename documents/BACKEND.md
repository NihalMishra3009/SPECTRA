# BACKEND — SPECTRA EW Scheduler (FastAPI)

## Stack
Python 3.11 · FastAPI · uvicorn · NumPy · Gymnasium · Stable-Baselines3 · PyTorch · pytest

## Run
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
# Swagger UI:  http://localhost:8000/docs
```

## Tests
```powershell
.\.venv\Scripts\python -m pytest tests -q
```
**49 passing** = `test_engine.py` (12) + `test_api.py` (18: full REST surface +
WebSocket stream + validation) + `test_db.py` (6: SQLite persistence) +
`test_dataset_rfi.py` (8) + `test_rfi_ucb.py` (5: friend's RandomForest
integration). API tests cover every endpoint, `POST /api/simulate` across all
14 scenarios **and** all 10 schedulers, deterministic-seed replay, 400/422
error handling, `/ws/simulate` streaming + bad-payload fallback, and both
external-model contracts.

## Module map

```
app/
├─ main.py              FastAPI app + CORS (alone)
├─ config.py            SimConfig (pydantic) | SCHEDULERS | DEFAULT_DEMO
├─ api/routes.py        REST + WebSocket endpoints
├─ sim/
│  ├─ emitter.py        Emitter: fixed / periodic / hopping / bursty / threat_weight
│  ├─ environment.py    RFEnvironment: ground-truth (T×B) bool grid + events + active segments
│  ├─ receiver.py       Receiver: scans 1 band → (hit, snr) ; optional Pfa noise
│  ├─ scenarios.py      all 12 PS scenarios + stable_switch_surprise + periodic_only presets
│  ├─ metrics.py        PS figures of merit (burst-segmented interception)
│  ├─ engine.py         run_simulation / make_scheduler / run_writer (WS generator)
│  ├─ gym_env.py        EWGymEnv — Gymnasium env for SB3 training
│  └─ schedulers/
│     ├─ baseline.py    BaseScheduler + RoundRobin
│     ├─ bandit.py      EpsilonGreedy · UCB1 · ThompsonSampling (EWMA recency + sweep floor)
│     ├─ adaptive.py    SlidingWindow (windowed recency)
│     ├─ rl_policy.py   RlPolicyScheduler — serves SB3 artifact + build_obs()
│     ├─ sequence.py    SequenceScheduler — serves LSTM timing/hop predictor
│     ├─ dataset_rfi.py DatasetRFIScheduler — serves an external dataset-trained model
│     └─ rfi_ucb.py     RFIUCBScheduler — friend's RandomForest prior blended into UCB1
├─ train/               Offline training scripts (see MODEL.md) + artifacts/
│   ├─ rfi_model.py     importable demo model class (pickle-safe)
│   └─ integrate_dataset_model.py  drop-in CLI for your trained model (see MODEL_CONTRACT.md)
└─ data/turing_loader.py  HF Turing radar dataset seam (Phase 5)
```

## Key ideas

- **Fair comparison** — `run_simulation` runs the baseline *and* the smart
  scheduler on the **same seeded environment**; everything is deterministic
  (seeded RNGs) so replays are bit-identical.
- **Interception is burst-segmented** — a transmission is one `(band, start, end)`
  segment; intercepted = first hit inside the segment. Interception ratio =
  hit segments / total segments. This is the honest EW definition (not per-cell).
- **Real frequency grid** — each abstract band maps to a real GHz sub-range
  (see *Real frequency mapping* below); the emitter's band = its operating freq.
- **Adaptation speed** = time steps from the first ground-truth *change* event to
  the smart scheduler's first hit on a newly-active band.
- **Scheduler protocol**: `tick(t) → select(t) → update(band, hit, t)`.
  Bandits decay all band estimates by `(1-α)` per step; a hit raises that band by
  `α`. The **sweep floor** (`floor`, default 0.15) guarantees broad coverage.

## API

| Endpoint | Purpose |
|---|---|
| `GET  /api/health` | health |
| `GET  /api/scenarios` | 14 scenario presets |
| `GET  /api/schedulers` | 7 schedulers + demo defaults |
| `GET  /api/models` | trained artifacts + meta |
| `GET  /api/curves/<file>` | training curves JSON |
| `POST /api/simulate` | full run JSON (replay-able) |
| `WS   /ws/simulate` | per-step live telemetry stream |
| `GET /api/db/runs` | recorded runs (SQLite) |
| `GET /api/db/runs/{run_id}` | full stored run |
| `GET /api/db/stats` | persistence totals + best run |

`POST /api/simulate` validates the request: an unknown `scenario` or
`scheduler` returns **`400`** with a clear detail message; out-of-range numeric
fields return **`422`** (pydantic). Unknown scenario IDs used to silently fall
back to the default preset — now they are rejected.

`POST /api/simulate` body (SimConfig):
```json
{ "n_bands": 10, "n_steps": 300, "seed": 2024, "scenario": "stable_switch_surprise",
  "scheduler": "thompson", "alpha": 0.25, "epsilon": 0.05, "window": 40, "floor": 0.15,
  "freq_start_ghz": 2.0, "freq_end_ghz": 18.0 }
```

### Real frequency mapping (sender side)

Bands are abstract indices *until* the frequency grid maps them to RF sub-ranges:

- **`freq_start_ghz` / `freq_end_ghz`** — the guarded spectrum (default 2–18 GHz).
- Band `i` → GHz range `[start + i·w, start + (i+1)·w]`, `w = span/n_bands`.
- The emitter (sender) occupies one band → its operating frequency = that band's
  range (e.g. band 7 → **13.2–14.8 GHz**, center 14.0 GHz for 2–18/10).
- Results include **`band_edges_ghz`** (`[[lo,hi],…]` per band) so the UI can
  label the waterfall Y-axis and spectrum map with real GHz.

Add a scheduler → add the class in `app/sim/schedulers/`, register it in
`app/config.py:SCHEDULERS` and `engine.make_scheduler`. Add a scenario → add an
`Emitter` factory in `app/sim/scenarios.py` and a catalog entry.