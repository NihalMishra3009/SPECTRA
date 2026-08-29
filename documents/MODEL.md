# MODELS — Training Pipeline (offline · local · CPU · free)

Everything trains on your machine in minutes and is saved to
**`backend/app/train/artifacts/`**. No GPU, no cloud needed.

## Artifacts registry

| Artifact | File(s) | Type |
|---|---|---|
| DQN | `dqn.zip` + `dqn_meta.json` + `dqn_curves.json` | Stable-Baselines3 `MlpPolicy` |
| PPO | `ppo.zip` + `ppo_meta.json` + `ppo_curves.json` | Stable-Baselines3 `MlpPolicy` |
| Bandit baseline | `bandit_baseline.json` | catalog evals (fairness reference) |
| Sequence LSTM | `sequence.pt` + `meta.json` + `sequence_curves.json` | PyTorch LSTM (next-band + quiet) |

## Train (from `backend/`)

```powershell
.\.venv\Scripts\python -m app.train.train_dqn      --steps 40000
.\.venv\Scripts\python -m app.train.train_ppo      --steps 40000
.\.venv\Scripts\python -m app.train.train_sequence --epochs 60
.\.venv\Scripts\python -m app.train.train_bandit
```

> Trained artifacts are already committed in `app/train/artifacts/` — the demo
> runs with zero training.

## RL observation space (shared train ⇄ inference)

`obs = [ EWMA activity (n) , normalised counts (n), normalised time (1) ]` → dim `2n+1`.
Built identically by `EwgymEnv.step()` and `schedulers/rl_policy.build_obs()`.

- **Action** = band to scan next (`Discrete(n)`).
- **Reward** = `+1` on hit, `-0.5` on miss (matches the engine's reward/cost).
- **Env** = `EWGymEnv` draws each episode from a training palette
  (`stable, switch, surprise, hopper, burst, multi_sim, periodic, random`)
  with a fresh seed so policies generalise instead of memorising.

## Sequence model (LSTM) — `SeqModel` in `app/train/seq_model.py`

Predicts the emitter's **next active band** (+ a *quiet* class) from the last
`window=20` scan observations. Feature per step = `[one-hot band (n), active, quiet]`.
Used by the `sequence` scheduler: confident predictions bias the scan; otherwise
it falls back onto the EWMA activity estimate.

## Current results (10 bands)

| Model | Metric |
|---|---|
| Sequence LSTM | val acc **0.949 → 0.954** |
| DQN (40k steps) | eval interception **5.5% → 18.3%** over training |
| PPO (40k steps) | eval interception **→ 31.3%**, reward → 50/episode |
| Bandits (catalog mean) | thompson **66.0%** · ucb1 65.3% · ε-greedy 63.5% · window 59.5% |

### Live interception (smart vs open-loop baseline)
- `sequence` on **periodic_only**: **47.1% vs 7.3%** (6.5×)
- `sequence` on **hopper**: **24.5% vs 10.6%** (2.3×)
- stable/switch/surprise: smart matches coverage on all, reward +0.6 → +0.9
  higher than baseline's −0.35, adaptation detected in 2–31 steps.

## Registry & serving

`GET /api/models` lists every artifact with its `meta.json`; `GET /api/curves/*`
streams training curves to the dashboard's Model panel. To switch the live
scheduler: pick it in the dashboard (ConfigPanel / ModelPanel) — the engine
loads the SB3/LSTM artifact on demand.

## Notes
- Training is **offline & deterministic** (`--seed`); artifacts ship with
  `meta.json` (hyperparams, palette, wall time) for reproducibility.
- Phase 5 (roadmap): validate against the **Turing Synthetic Radar Dataset**
  (`backend/data/turing_loader.py` seam), then gradient/RL upgrades stay local.