"""Train the sequence LSTM that predicts periodic/hopping emitter timing.

Learns, from raw past observations, the *next band* an emitter will be on
(plus a 'quiet' class when nothing fires). Saved to:
    artifacts/sequence.pt · artifacts/meta.json · artifacts/sequence_curves.json

Usage:
    python -m app.train.train_sequence [--epochs 80]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..sim.emitter import Emitter
from . import ARTIFACTS_DIR
from .seq_model import SeqModel


def gen_active_sequence(seed: int, n_bands: int = 10, steps: int = 80) -> np.ndarray:
    """Simulate a periodic / hopping / mixed emitter and record active band per step."""
    rng = np.random.default_rng(seed + 123)
    active = np.full(steps, -1, dtype=int)
    kind = int(rng.integers(3))
    if kind == 0:
        em = Emitter("t", [int(rng.integers(n_bands))], interval=int(rng.integers(3, 13)),
                     phase=int(rng.integers(12)), rng_seed=seed)
        for t in range(steps):
            b = em.active_band_at(t)
            if b is not None:
                active[t] = b
    elif kind == 1:
        k = int(rng.integers(2, min(n_bands, 5)))
        em = Emitter("t", [int(rng.integers(n_bands))], interval=int(rng.integers(3, 13)),
                     phase=int(rng.integers(12)),
                     hop_bands=rng.choice(n_bands, size=k, replace=False).tolist(),
                     rng_seed=seed)
        for t in range(steps):
            b = em.active_band_at(t)
            if b is not None:
                active[t] = b
    else:
        ems = [
            Emitter("a", [int(rng.integers(n_bands))], interval=int(rng.integers(3, 9)),
                    phase=int(rng.integers(8)), rng_seed=seed + 1),
            Emitter("b", [int(rng.integers(n_bands))], interval=int(rng.integers(3, 9)),
                    phase=int(rng.integers(8)), rng_seed=seed + 2),
        ]
        flip = np.random.default_rng(seed + 3)
        for t in range(steps):
            on = [e.active_band_at(t) for e in ems]
            on = [b for b in on if b is not None]
            if on:
                active[t] = int(flip.choice(on))
    return active


def build_dataset(n_episodes: int, n_bands: int, window: int, steps: int, seed: int):
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    for ep in range(n_episodes):
        active = gen_active_sequence(seed + ep * 37, n_bands, steps)
        hist: list[np.ndarray] = []
        for t in range(steps):
            a = active[t]
            onehot = np.zeros(n_bands, dtype=float)
            act, quiet = 0.0, 1.0
            if a >= 0:
                onehot[a] = 1.0
                act, quiet = 1.0, 0.0
            hist.append(np.concatenate([onehot, np.array([act, quiet])]))
        for t in range(window, steps):
            X = np.stack(hist[t - window : t])
            y = int(active[t]) if active[t] >= 0 else n_bands  # quiet class index
            rows_X.append(X)
            rows_y.append(y)
    return np.stack(rows_X), np.array(rows_y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-bands", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    X, y = build_dataset(args.episodes, args.n_bands, args.window, 80, args.seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = Xt.shape[0]
    n_tr = int(n * 0.8)
    tr = TensorDataset(Xt[:n_tr], yt[:n_tr])
    va = TensorDataset(Xt[n_tr:], yt[n_tr:])
    tr_dl = DataLoader(tr, batch_size=128, shuffle=True)
    va_dl = DataLoader(va, batch_size=256)

    model = SeqModel(args.n_bands, args.window)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    best_acc = 0.0
    curves: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        total, correct, run_loss = 0, 0, 0.0
        for xb, yb in tr_dl:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.shape[0]
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.shape[0]
        model.eval()
        v_cor = v_tot = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                out = model(xb)
                v_cor += (out.argmax(1) == yb).sum().item()
                v_tot += xb.shape[0]
        acc = v_cor / max(1, v_tot)
        best_acc = max(best_acc, acc)
        curves.append({"epoch": epoch + 1, "train_acc": round(correct / max(1, total), 4),
                       "val_acc": round(acc, 4), "epoch_loss": round(run_loss / max(1, total), 4)})
    torch.save(model.state_dict(), ARTIFACTS_DIR / "sequence.pt")
    meta = {
        "model": "lstm",
        "n_bands": args.n_bands,
        "window": args.window,
        "layers": 1,
        "hidden": 64,
        "epochs": args.epochs,
        "episodes": args.episodes,
        "seed": args.seed,
        "threshold": 0.55,
        "final_val_acc": round(best_acc, 4),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (ARTIFACTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS_DIR / "sequence_curves.json").write_text(json.dumps(curves, indent=2))
    print(f"[sequence] val acc {best_acc:.3f} in {meta['wall_seconds']}s -> artifacts/sequence.pt")


if __name__ == "__main__":
    main()