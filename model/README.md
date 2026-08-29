# model/ — all trained models + training outputs in one place

Repository-level folder that centralizes every ML artifact and evaluation
output of the SPECTRA project, so you don't have to dig through code.

| Path | What's inside |
|---|---|
| `model/artifacts/` | trained models (DQN, PPO, LSTM, dataset model) + metadata + training curves |
| `model/MODELS_INVENTORY.md` | one-line table of every artifact, what it does, training command, result |
| `backend/app/train/` | the *training code* (scripts stay with the app for serving) |
| `documents/MODEL_CONTRACT.md` | how to drop in an externally trained model |

> Code that *serves* the models (loading + inference) lives in
> `backend/app/train/` and `backend/app/sim/schedulers/`; the **files** they
> serve all live here.

## Where all model files are (gitignored none — all committed)

```
model/
└── artifacts/            ← THE one place every trained model lives
    ├── dqn.zip           SB3 DQN policy
    ├── ppo.zip           SB3 PPO policy
    ├── sequence.pt       LSTM next-band timing/hop predictor
    ├── turing_model.pkl  dataset-trained emitter model (drop-in via MODEL_CONTRACT)
    ├── bandit_baseline.json
    ├── *_meta.json       reproducibility metadata (seed, hyperparams)
    └── *_curves.json     training curves (dashboard sparklines)
```

## How it's wired

- `backend/app/train/__init__.py::ARTIFACTS_DIR` = `model/artifacts`
  (override for deployed envs with `SPECTRA_MODEL_DIR` / `SPECTRA_ARTIFACTS_DIR`).
- `GET /api/models` lists exactly these files with their meta.
- Dashboard Model panel shows them + curves, and lets you swap any in as the
  live smart brain.

## Retrain / rebuild an artifact

```powershell
cd backend
.\.venv\Scripts\python -m app.train.train_dqn      # -> model/artifacts/dqn.zip
.\.venv\Scripts\python -m app.train.train_ppo      # -> model/artifacts/ppo.zip
.\.venv\Scripts\python -m app.train.train_sequence # -> model/artifacts/sequence.pt
.\.venv\Scripts\python -m app.train.train_bandit   # -> model/artifacts/bandit_baseline.json
.\.venv\Scripts\python -m app.train.integrate_dataset_model --model path\\friend_model.pkl
```