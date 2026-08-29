# SPECTRA — Smart Scan Strategy for Electronic Warfare

**SIH Problem 26055 · DRDO / IDEX · ML-based ES Receiver Scheduler**

A closed-loop, machine-learning scan scheduler for a bandwidth-limited Electronic
Support receiver. It observes real-time **hits/misses**, learns which bands are
active, adapts when emitter behaviour changes, and decides where the receiver
should look next — beating the traditional open-loop round-robin sweep.

![stack](https://img.shields.io/badge/Python-3.11-blue) ![torch](https://img.shields.io/badge/PyTorch-black) ![sb3](https://img.shields.io/badge/Stable--Baselines3-green) ![react](https://img.shields.io/badge/React-19-cyan) ![taiwind](https://img.shields.io/badge/Tailwind-v4-blue)

---

## Demo (one command — everything: database + backend + dashboard)

```powershell
# from the repo root — installs deps if needed, seeds database/spectra.db,
# starts FastAPI on :8000 and Vite on :5173
.\run.ps1
```

Or run the three parts manually (separate terminals):

```powershell
# 1) Database  (auto-created on first run; seed sample history to populate the UI)
cd backend
.\.venv\Scripts\python -m app.seed_db

# 2) Backend  (http://localhost:8000)
cd backend
python -m venv .venv                  # first time only
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn app.main:app --port 8000

# 3) Dashboard  (http://localhost:5173)
cd frontend
npm install                           # first time only
npm run dev
```

Open **http://localhost:5173**, press **▶ RUN SIMULATION** (or **▶ DEMO MODE**
for the 4-scenario guided script) and watch the animated waterfall.

> Old trained models are already in `backend/app/train/artifacts/` — no training
> needed for the demo.
>
> **No backend? No worries.** Use **⚡ LOAD MOCK DATA** in the config panel — a
> bundled offline fixture (`frontend/public/mock/demo.json`) replays a full run.
> The dashboard also auto-falls back to mock if the API is unreachable.

---

## What it does

```
RF env (ground truth) → receiver scans 1 band → hit/miss → scheduler updates
                        band priorities → picks next band → (closed loop)
```

- **Baseline** (`round_robin`): the open-loop fixed sweep used by legacy systems.
- **Smart schedulers**: `epsilon_greedy` · `ucb1` · `thompson` · `rl_dqn` ·
  `rl_ppo` (Stable-Baselines3) · `sequence` (LSTM timing/hop predictor).
- **Adaptation**: EWMA recency decay (α) — old evidence fades, pattern switches
  re-prioritize the scheduler in a few steps.
- **Exploration floor**: guaranteed broad-sweep coverage keeps surprise emitters
  detectable without sacrificing focus (Scenario 5 → falls back to uniform).

## Metrics (all 7 PS figures of merit)

Interception ratio · Avg intercept time · Probability of detection · Probability
of false alarm · Miss count · Adaptation speed · Reward/cost (avg reward).

## Measured results (seed 42, 10 bands · 300 steps)

| Scenario pair | Baseline | Smart | Winner |
|---|---|---|---|
| Stable (whole-run bands) | 100% | 100% | reward −0.35 → **+0.65** |
| Pattern switch | 100% | 100% | adaptation ≈ 6–31 t |
| Surprise emitter | 100% | 100% | detected +2–15 t |
| Periodic timing (pure) | 7.3% | **47.1%** (sequence LSTM) | 6.5× |
| Frequency hopper | 10.6% | **24.5%** (sequence) | 2.3× |

Plus: LSTM **94.9–95.4%** next-emitter accuracy · Bandit catalog mean
interception: thompson 66.0% / ucb1 65.3% / ε-greedy 63.5%.

---

## Repository layout

```
backend/          FastAPI + simulator + training pipeline (see documents/BACKEND.md)
frontend/         React 18 + Vite + Tailwind dashboard (see documents/FRONTEND.md)
database/         SQLite schema + recorded run measurements (spectra.db at runtime)
documents/        Submission report, problem statement, all project docs
```

## Docs

- **[documents/PROJECT_DOCUMENT.md](documents/PROJECT_DOCUMENT.md)** — the submission report (v2)
- **[documents/WORKFLOW.md](documents/WORKFLOW.md)** — entire software flow
- **[documents/BACKEND.md](documents/BACKEND.md)** — architecture, API, adding scenarios/schedulers, tests
- **[documents/FRONTEND.md](documents/FRONTEND.md)** — dashboard structure, theme, playback, deploy
- **[documents/MODEL.md](documents/MODEL.md)** — training pipeline, results, retrain commands

## Validation

```powershell
cd backend
.\.venv\Scripts\python -m pytest tests -q     # 30 tests
.\.venv\Scripts\python -m app.train.train_bandit  # bandit catalog table
```

## Deployment (free)

- **Backend** → HuggingFace Spaces (FastAPI) or Render free tier
- **Frontend** → Vercel/Netlify free tier
- Set `VITE_API_URL` in frontend to the deployed backend URL.