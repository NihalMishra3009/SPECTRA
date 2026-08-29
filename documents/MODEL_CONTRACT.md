# MODEL CONTRACT — integrating an externally trained emitter model

> **Friend's model is ALREADY integrated.** `model/artifacts/band_activity_model.pkl`
> (RandomForestClassifier, 4 features) runs live as scheduler **`rfi_ucb`** —
> UCB1 + ML prior. It needs `scikit-learn 1.9` + `joblib` + `pandas` (installed).
> This document describes the generic `dataset_rfi` contract too.

This is the exact interface your trained model (e.g. trained on the
**Turing Synthetic Radar Dataset** linked in the problem statement) must satisfy
to drop straight into SPECTRA as a live scan scheduler.

## 1. What the model must do

Given the receiver's scan history, output a **score per frequency band** —
how likely that band has an active emitter right now. The scheduler turns the
highest score into the next scan decision (with a safety exploration floor).

## 2. Input features (fixed size = `2 * n_bands + 1`)

One feature row per request, `dtype float`:

```
[ hit_ewma[0..n-1]               # per-band EWMA hit evidence (0..1)
, tanh(misses[0..n-1] / 5.0)     # per-band miss counts, squashed 0..1
, t_frac                         # t / total_steps  (0..1)
]
```

`n_bands` = number of frequency bands (default 10), 2–18 GHz grid. The exact
builder lives in `backend/app/sim/schedulers/dataset_rfi.py::build_features`.

## 3. Output

Any object providing **one** of:
- `model.predict_proba(X) -> (N, C)`   ← preferred (probabilities)
- `model.predict(X)        -> (N, C)`   (raw scores; first C columns used)

`X` is shape `(N, 2*n_bands+1)`, `C >= n_bands`. Column `j` = activation of band
`j` (higher = more likely active). We take `argmax` over the first `n_bands`.

## 4. File format

- **Recommended:** `pickle.dump(model, f)` — the model class must live in an
  **importable module** (not a notebook / `__main__`), so pickle can resolve it
  at load time.
- sklearn objects: `joblib.dump(model, path)` (joblib files are pickle-compatible;
  the loader tries pickle first, then joblib).

## 5. Drop-in steps

```powershell
cd backend
# put your file anywhere, e.g.  C:\models\my_radar_model.pkl
.\.venv\Scripts\python -m app.train.integrate_dataset_model --model C:\models\my_radar_model.pkl
```

The CLI: loads it, runs a quick switch-scenario eval, copies it to
`model/artifacts/turing_model.pkl`, writes `turing_model_meta.json`.

Then in the dashboard pick scheduler **`dataset_rfi`** → RUN.

## 6. Test your artifact first

```powershell
.\.venv\Scripts\python -c "from app.sim.schedulers.dataset_rfi import load_model, predict_scores; import numpy as np; from app.train import ARTIFACTS_DIR; m=load_model(ARTIFACTS_DIR/'turing_model.pkl'); print(predict_scores(m, np.zeros((1, 21))).shape)"
# expected: (1, 10)
```

## 7. Built-in demo (no model needed)

```powershell
.\.venv\Scripts\python -m app.train.integrate_dataset_model --build-demo
```
Trains a NumPy softmax next-band predictor on the built-in RF simulator → proves
the whole chain (`dataset_rfi` = green in dashboard Model panel) until your real
dataset-trained file replaces it.

## 8. Notes

- Missing/misbehaving artifact **never crashes the API** — the scheduler
  degrades to an epsilon-greedy floor.
- The feature vector is deliberately small & deterministic → any model
  (sklearn, torch, hand-rolled) can consume it without a feature forge refactor.