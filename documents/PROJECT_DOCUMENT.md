# PROJECT DOCUMENT · v2

# SPECTRA
## Smart Priority-based Electronic warfare Cognitive Tracking and Receiver Allocator
### ML-Driven Closed-Loop Scan Scheduler for Electronic Support Receivers

**Problem Statement 26055** — DRDO · Department of Defence Production / IDEX · Software · Robotics & Drones

**FastAPI** · **React + Vite** · **SB3 DQN/PPO** · **PyTorch LSTM** · **WebSocket Live**

---

## 01 · Executive Summary

SPECTRA is a closed-loop, machine-learning-based scan scheduler for Electronic Support (ES) receivers. It replaces the traditional fixed, open-loop sweep with a system that learns from real-time hit/miss feedback, adapts when emitter behaviour changes, and predicts where to scan next — improving interception rate and cutting intercept time, without relying on prior intelligence about emitters.

The system is fully built end-to-end: a physics-flavoured RF simulation engine, eight interchangeable scheduling algorithms (from simple bandits to trained Deep RL and LSTM sequence models), a FastAPI serving layer with REST and WebSocket streaming, and a live React dashboard with waterfall visualization, KPI cards, and scenario-driven demo mode.

| 14 Scenarios | 8 Schedulers | 7 PS Metrics | 30 Tests Passing | Live WebSocket Demo |
|---|---|---|---|---|

---

## 02 · Problem Statement

| FIELD | DETAIL |
|---|---|
| Problem ID | 26055 |
| Title | Smart Scan strategy for Electronic Warfare |
| Organization | DRDO — Department of Defence Production / IDEX |
| Category | Software |
| Theme | Robotics and Drones |

ES receivers must scan a wide frequency spectrum with instantaneous bandwidth an order of magnitude smaller than the total spectrum, forcing a sweep across bands. Open-loop, pre-mission scan plans waste time on inactive bands and can miss new or high-priority emitters. The objective is a closed-loop scheduler, trained on hits/misses, that maximizes interception rate and minimizes intercept time — including against periodic and frequency-agile emitters.

---

## 03 · System Architecture

The system separates cleanly into an **offline training plane** and a **live runtime plane**, connected through versioned model artifacts.

```
OFFLINE (training)                            LIVE RUNTIME

EWGymEnv -> SB3 DQN/PPO -> dqn.zip/ppo.zip  \   SimConfig -> Scenarios -> RFEnvironment
sequence data -> LSTM -> sequence.pt        --> artifacts/   -> make_scheduler -> {RoundRobin | Bandit | RL | Sequence}
scenario evals -> bandit_baseline.json      -/   loop t=0..T-1: tick -> select -> observe -> update
                                                    -> metrics (7 PS FOM) -> FastAPI (REST + WS) -> React dashboard
```

### 3.1 Backend Layout

| PATH | RESPONSIBILITY |
|---|---|
| `app/config.py` | SimConfig (pydantic) + SCHEDULERS registry — every run parameter |
| `app/sim/emitter.py` | Fixed / periodic / hopping / bursty / surprise emitter models |
| `app/sim/environment.py` | RFEnvironment — ground_truth matrix, events, GHz band edges |
| `app/sim/receiver.py` | Single scan receiver — 1 band probed per step, optional Pfa |
| `app/sim/scenarios.py` | 14 preset scenario factories mapped to PS emitter types |
| `app/sim/engine.py` | Control loop, run_simulation (baseline vs smart), make_scheduler |
| `app/sim/metrics.py` | compute_metrics — the 7 PS figures of merit |
| `app/sim/gym_env.py` | Gymnasium-compatible env used for SB3 training |
| `app/sim/schedulers/*.py` | baseline · bandit (ε-greedy/UCB1/Thompson) · adaptive · rl_policy · sequence |
| `app/train/*.py` | train_dqn, train_ppo, train_sequence, train_bandit, seq_model |
| `app/train/artifacts/` | dqn.zip, ppo.zip, sequence.pt, *_meta.json, *_curves.json, bandit_baseline.json |
| `app/api/routes.py` | REST + WebSocket endpoints |
| `data/turing_loader.py` | Turing Synthetic Radar Dataset integration seam (Phase 5) |
| `tests/test_engine.py` | 12 pytest cases — scenarios, schedulers, adaptation, priorities |
| `tests/test_api.py` | 18 pytest cases — every REST endpoint + WebSocket stream, validation |

### 3.2 Frontend Layout

| COMPONENT | ROLE |
|---|---|
| `useSim` (hook) | POST /api/simulate -> SimResult; playback state machine; offline mock fallback |
| `WaterfallChart` | Time × Band heatmap — truth signal, green HIT, red MISS, live sweep line |
| `ComparisonChart` | Cumulative interception % and avg reward, smart vs baseline |
| `KPIs` | PS metric cards with delta badges vs baseline |
| `AlertFeed` | Surprise/switch events with detection-delay log |
| `SpectrumMiniMap` | Per-band activity and smart-scanner hits, shown in GHz |
| `PlaybackControls` | Play/pause, scrub, 0.5x–4x speed |
| `ModelPanel` | Trained artifact status + curves; select model as live brain |
| `TopBar` | Scenario / scheduler / model selectors; Demo Mode toggle |

---

## 04 · Scheduling Algorithms

| SCHEDULER | MECHANISM |
|---|---|
| `round_robin` | Open-loop fixed sweep — baseline, no learning |
| `epsilon_greedy` | EWMA recency estimate + ε exploration + minimum sweep floor |
| `ucb1` | Upper Confidence Bound bandit, same decay + floor |
| `thompson` | Thompson Sampling bandit, same decay + floor |
| `adaptive_window` | Sliding-window recency estimate |
| `rl_dqn` / `rl_ppo` | Stable-Baselines3 policy inference from trained dqn.zip / ppo.zip |
| `sequence` | PyTorch LSTM (sequence.pt) predicting next active band and timing |

SB3 and torch are imported lazily so the API remains fast to boot even when trained artifacts are absent.

---

## 05 · Scenario Coverage

14 preset scenarios are built from five emitter primitives, matching every behaviour pattern the PS requires the scheduler to handle:

| EMITTER TYPE | BEHAVIOUR | EXAMPLE SCENARIOS |
|---|---|---|
| Fixed | Always-on band(s) | `stable`, `multi_sim` |
| Periodic | On every `interval` steps, phase φ | `periodic`, `periodic_only` |
| Hopping | Band changes every `hop_period` | `hopper`, `threat` |
| Bursty | On/off with a duty cycle | `burst` |
| Surprise | Silently activates at a later start step | `surprise` |

---

## 06 · Evaluation Metrics — 7 PS Figures of Merit

| # | METRIC | DEFINITION |
|---|---|---|
| 1 | Interception Ratio | Burst-segmented; first hit per (band,start,end) segment wins |
| 2 | Avg Intercept Time | Mean first-hit delay across segments |
| 3 | Probability of Detection | Likelihood an active emitter is detected |
| 4 | Probability of False Alarm | Likelihood of a false positive detection |
| 5 | Miss Count | Transmissions never detected in the run |
| 6 | Adaptation Speed | Steps to first hit after a ground-truth change event |
| 7 | Avg Reward / Cost | Cumulative reward: hit +1, miss −0.5 |

---

## 07 · Offline Training Pipeline

| MODULE | WHAT IT TRAINS | ARTIFACT |
|---|---|---|
| `gym_env.py` | EWGymEnv — reward +1 hit / −0.5 miss, action=band, obs=[activity, counts, time] | — |
| `train_dqn.py` | SB3 DQN, MlpPolicy | dqn.zip, dqn_meta.json, dqn_curves.json |
| `train_ppo.py` | SB3 PPO, MlpPolicy | ppo.zip, ppo_meta.json, ppo_curves.json |
| `train_sequence.py` + `seq_model.py` | PyTorch LSTM — next-band + quiet class | sequence.pt, meta.json, sequence_curves.json |
| `train_bandit.py` | Eval catalog of the 4 non-RL schedulers | bandit_baseline.json |

Training is offline and seeded; artifacts ship inside the repository so the live demo does not require retraining.

---

## 08 · Serving Layer — API Surface

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | service status |
| GET | `/api/scenarios` | preset list for the dashboard picker |
| GET | `/api/schedulers` | scheduler registry |
| GET | `/api/models` | artifact presence + metadata |
| GET | `/api/curves/<f>` | training curves for sparklines |
| POST | `/api/simulate` | deterministic full run -> SimResult JSON |
| WS | `/ws/simulate` | per-step telemetry stream |

---

## 09 · Validation & Test Coverage

- **pytest — 30 cases**: scenario generation, bandit beats baseline on stable pattern, periodic capture via Thompson Sampling, surprise detection, adaptation speed, priority trajectory, plus a full API test suite (health, scenarios, schedulers, models, curves, simulate across all 8 schedulers & all 14 scenarios, deterministic-seed replay, 400 validation, 422 bounds, WebSocket streaming + fallback).
- **train_bandit catalog** — Thompson Sampling leads at **66.0%** interception vs ≈**59.5%** for weaker baselines on the evaluated scenario set.
- **Offline mock fixture** — `frontend/public/mock/demo.json` (92 KB deterministic run) lets the dashboard demo without a live backend.

---

## 10 · Demo Walkthrough

1. Dashboard **RUN** triggers `POST /api/simulate` with the selected SimConfig.
2. Backend builds scenario → environment → runs baseline and smart loop on identical ground truth → computes metrics.
3. JSON result returns; dashboard animates the tiled waterfall and fires alerts on scenario events.
4. Model panel surfaces trained DQN / PPO / LSTM curves; any can be swapped in as the live “smart” brain.
5. If the backend is unavailable, the mock fixture replays an identical run fully offline.

**Demo Mode** auto-plays four scenarios back to back — stable → surprise → periodic (LSTM) → hopper — at 2× speed, giving a self-running showcase for evaluators.

---

## 11 · Future Scope

- Full integration of the **Turing Synthetic Radar Dataset** (scan-mode PDWs) via `data/turing_loader.py` for real-world validation.
- **Multi-receiver** and spatially-distributed scan coordination.
- Noise/false-alarm modelling refinement for tighter Pfa accuracy.
- **Threat-priority weighting** layered on top of raw activity-based scheduling.

---

*SPECTRA · End of Document*