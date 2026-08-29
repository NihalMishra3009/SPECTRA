# WORKFLOW — SPECTRA Smart Scan Strategy (full software flow)

This document traces the **entire system**: how data is produced, how the ML
models are trained, how a live run is simulated, and how the dashboard shows it.

---

## 0. Big picture (once sentence)

*"Generate enemy RF truth → a bandwidth-limited receiver probes ONE band per step
→ the smart scheduler learns from its hits/misses → predicts the next band to
tune → dashboard replays everything with metrics vs the open-loop baseline."*

```
┌────────────────────────────── OFFLINE (training) ──────────────────────────────┐
│  EWGymEnv ──► SB3 DQN/PPO ──► dqn.zip / ppo.zip ──┐                             │
│  sequence data  ──► LSTM ──► sequence.pt ────────┼──► artifacts/ (+meta.json)  │
│  scenario evals  ──► bandit_baseline.json ──────┘                              │
└────────────────────────────────────────────────────────────────────────────────┘
                                        │ loaded on demand
                                        ▼
┌────────────────────────────── LIVE RUNTIME ─────────────────────────────────────┐
│ SimConfig ─► Scenarios ─► RFEnvironment ─► (baseline + smart both run on same   │
│    │            │             │  ground_truth/events/bands/segments             │
│    ▼            ▼             ▼                                                 │
│ make_scheduler ◄ SCHEDULERS  RoundRobin  vs  Bandit/RL/Sequence                │
│                                    │                                            │
│                          loop t=0..T-1:                                         │
│                     tick(t) → select(band) → receiver.observe → (hit,snr)       │
│                                    └────► update(band,hit,t)   ◄── learn          │
│                                    │                                            │
│                                    ▼   log[] + metrics (PS 7 FOM)               │
│                          FastAPI  POST /api/simulate  / WS per-step              │
│                                    ▼                                            │
│                          Dashboard playback + waterfall + KPIs + alerts         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Configuration entry — `app/config.py`

`SimConfig` (pydantic) defines every knob:

```
n_bands=10 · n_steps=300 · seed · scenario · scheduler
alpha (decay) · epsilon (explore) · window · floor (sweep coverage)
freq_start_ghz=2.0 · freq_end_ghz=18.0   ← real RF grid for the bands
```

- `SCHEDULERS` — registry: `round_robin, epsilon_greedy, ucb1, thompson,
  adaptive_window, rl_dqn, rl_ppo, sequence` (+ each one's demo defaults).
- A REST/WS request carries exactly this object → deterministic run.

## 2. Scenario build — `app/sim/scenarios.py`

Each of the **14 presets** is a factory returning `Emitter[]`.
Emitter types = the PS emitters:

| Emitter kind | Models | Used by scenario |
|---|---|---|
| fixed (steady) | always-on band(s) | stable, multi_sim |
| periodic | every `interval` steps, phase `φ` | periodic, periodic_only |
| hopping | band changes per `hop_period` | hopper, threat |
| bursty | on/off with duty cycle | burst |
| surprise | turns on at `start` silently | surprise |

`get_emitters(scenario, n, steps)` is the only bullet in → environment.

## 3. Environment — `app/sim/environment.py`

`RFEnvironment` builds the **truth model** (PS: "status of each band, each time
slot = transmission / non-transmission"):

- `ground_truth` : `(T × B)` boolean matrix — emitted once, used by BOTH runtimes.
- `events[]`     : `{t, type: change|surprise, bands_on|off}` — the adaptation hooks.
- `activity_profile[]`, `active_segments[]`, `total_transmissions`, `n_segments`.
- `band_edges_ghz[]` — real RF sub-ranges per band (2–18 GHz / 10 → 1.6 GHz each).

## 4. Scheduler selection — `app/sim/engine.py :: make_scheduler`

```
round_robin     → open-loop sweep (baseline)                [no learning]
epsilon_greedy  → EWMA recency + ε exploration + sweep floor
ucb1 / thompson → bandits with same decay + floor
adaptive_window → sliding-window recency
rl_dqn/rl_ppo   → loads SB3 artifact (dqn.zip/ppo.zip), obs=building 2n+1
sequence        → loads LSTM (sequence.pt), predicts next band timing
```

SB3/torch are imported lazily so the API boots fast even without artifacts.

## 5. The control loop — `app/sim/engine.py :: _run_single` (core)

For each `t` in `0..T-1`:

```
scheduler.tick(t)                 → decay all band estimates by (1-α)
band   = scheduler.select(t)      → MODEL PREDICTION: which band to tune
hit,snr = receiver.observe(band,t)→ hardware probe (1/10 of spectrum)
scheduler.update(band,hit,t)      → reward: hit += α  ·  miss decays
log.push( t, band, hit, snr, reward, cumulative_ratio )
```

`receiver.py::Receiver` = the PS's high-sensitivity / order-lower-IBW sweeper
(sees exactly 1 band, tunable, optional Pfa). One receiver, one answer.

`run_simulation` runs **RoundRobin on the same environment** then the smart
scheduler → apples-to-apples comparison; also records smart's per-step priority
(q vector) for the belief heatmap.

## 6. Metrics — `app/sim/metrics.py`

`compute_metrics` implements the **7 PS figures of merit**:

1. Interception ratio — burst-segmented (segment `(band,start,end)`, first hit wins)
2. Avg intercept time (mean first-hit delay over segments)
3. Probability of detection
4. Probability of false alarm
5. Miss count
6. Adaptation speed (steps to first hit after a ground-truth *change* event)
7. Avg reward / cost (hit +1, miss −0.5)

## 7. Offline training pipeline — `app/train/`

| Module | What | Artifact |
|---|---|---|
| `gym_env.py` | `EWGymEnv` — Gymnasium env, reward `+1 hit / −0.5 miss`, action = band, obs `[activity, counts, time]`, per-episode random palette + seed | — |
| `train_dqn.py` | SB3 DQN `MlpPolicy` | `dqn.zip`, `dqn_meta.json`, `dqn_curves.json` |
| `train_ppo.py` | SB3 PPO `MlpPolicy` | `ppo.zip`, `ppo_meta.json`, `ppo_curves.json` |
| `train_sequence.py` + `seq_model.py` | PyTorch LSTM → predicts next band (+ quiet class) | `sequence.pt`, `meta.json`, `sequence_curves.json` |
| `train_bandit.py` | eval catalog of the 4 non-RL smart schedulers | `bandit_baseline.json` |

Training is **offline & seeded** → artifacts ship in the repo, `FunctionRegistry`
declares which train script creates which file (`app/train/__init__.py`).

## 8. Serving layer — `app/api/routes.py`

```
GET  /api/health        status
GET  /api/scenarios     presets (dashboard picker)
GET  /api/schedulers    scheduler registry
GET  /api/models        artifact presence + meta
GET  /api/curves/<f>    training curves (sparklines)
POST /api/simulate      deterministic full run → SimResult JSON
WS   /ws/simulate       per-step telemetry stream generator
```

`run_writer` wraps the same loop (Section 5) as an async generator — the WebSocket
ticks one message per step.

## 9. Frontend — `frontend/src`

```
useSim (hook)            run() → POST /api/simulate → SimResult
                         playback: cur + speed tick ; offline → mock fallback
WaterfallChart           heatmap: truth≡signal / green HIT / red MISS + sweep line
ComparisonChart          cumulative interception % + avg reward, smart vs baseline
KPIs                     PS metrics cards + Δ badges
AlertFeed                surprise/switch events → detection delay log
SpectrumMiniMap          activity per band + smart hits (GHz ranges)
PlaybackControls         play/pause · scrub · 0.5–4×
ModelPanel               artifacts + training curves → "USE AS SMART"
TopBar                   scenario/scheduler/model chips · DEMO MODE
```

Demo Mode auto-plays 4 scenarios (stable→surprise→periodic LSTM→hopper) at 2×.

## 10. Validation

```
pytest            → 30 tests (12 engine + 18 API: all REST endpoints, all 14
                     scenarios × all 8 schedulers simulate, deterministic replay,
                     400 validation, 422 bounds, WebSocket stream + fallback)
train_bandit      → catalog table (thompson 66.0% > … 59.5%)
mock fixture      → frontend/public/mock/demo.json (92 KB deterministic run)
```

## TL;DR ordering for a demo

1. Dashboard RUN → **POST /api/simulate** (config). 
2. `scenarios → environment → (baseline vs smart loop) → metrics`.
3. JSON returns; dashboard animates tiled waterfall; alerts fire on events.
4. Model panel shows trained DQN/PPO/LSTM curves — pick one as the live brain.
5. No backend? Mock fixture replays the same run offline.