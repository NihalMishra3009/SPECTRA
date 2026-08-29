# Model Inventory — every trained artifact at a glance

| Artifact | Type | Trains | Command | Result |
|---|---|---|---|---|
| `dqn.zip` | SB3 DQN (MlpPolicy) | next-band scan policy on EWGymEnv | `python -m app.train.train_dqn --steps 200000` | stable **100%** · eval curve 5.5→18.3% |
| `ppo.zip` | SB3 PPO (MlpPolicy) | next-band scan policy on EWGymEnv | `python -m app.train.train_ppo` | stable **100%** · eval **31.3%** |
| `sequence.pt` (+meta) | PyTorch LSTM | next active band + quiet class (periodic/hop timing) | `python -m app.train.train_sequence --epochs 40` | val acc **95.4%**; periodic interception **47% vs 7%** |
| `turing_model.pkl` (+meta) | dataset-trained emitter model (demo: NumPy softmax) | next-band predictor from scan history | `python -m app.train.integrate_dataset_model --build-demo` | switch **100%** (real model = replace this file) |
| `band_activity_model.pkl` (+meta) | **friend's RandomForest (TSRD, 200 trees)** | per-band P(active next window) → UCB1 prior | copied from `Spectra_Model/models/` | acc **93.7%**, recall 0.81 |
| `bandit_baseline.json` | measurements | catalog eval of ε-greedy/UCB1/Thompson/window | `python -m app.train.train_bandit` | thompson **66.0%** · ucb1 65.3% · ε-greedy 63.5% |
| `dqn_meta.json` / `dqn_curves.json` | metadata | hyperparams, training palette, curves | auto | reproducibility |
| `ppo_meta.json` / `ppo_curves.json` | metadata | same for PPO | auto | — |
| `sequence_curves.json` | metadata | per-epoch train/val accuracy | auto | — |
| `turing_model_meta.json` | metadata | integration source + eval | auto | — |

## Scheduler ↔ artifact mapping (which "smart brain" uses which file)

```
round_robin      -> none (baseline, no learning)
epsilon_greedy / ucb1 / thompson / adaptive_window -> none (online bandits)
rl_dqn           -> model/artifacts/dqn.zip
rl_ppo           -> model/artifacts/ppo.zip
sequence         -> model/artifacts/sequence.pt (+ meta.json)
dataset_rfi      -> model/artifacts/turing_model.pkl   ← generic drop-in
rfi_ucb          -> model/artifacts/band_activity_model.pkl  ← FRIEND'S MODEL (live)
```

## Replace friend's dataset-trained model (5 min)

```powershell
cd backend
.\.venv\Scripts\python -m app.train.integrate_dataset_model --model C:\path\my_model.pkl
# -> copies to model/artifacts/turing_model.pkl + writes meta + quick eval
# dashboard -> scheduler `dataset_rfi` -> RUN
```

Full input/output spec: `documents/MODEL_CONTRACT.md`.