"""Dataset-trained RFI scheduler.

Integrates an EXTERNALLY trained emitter model (e.g. trained on the Turing
Synthetic Radar Dataset that ships with the problem statement) into the scan
loop. The model sees a fixed feature vector per step and returns per-band
activation scores; the scheduler turns those scores into a scan decision with
the usual exploration floor so nothing starves.

Model contract (see documents/MODEL_CONTRACT.md)
------------------------------------------------
The artifact is any pickle-able Python object implementing either
    model.predict(X)          -> (N, C) scores            (first C columns used)
    model.predict_proba(X)    -> (N, C) probabilities
where X is 2-D float32 of shape (N, 2*n_bands + 1): feature per row is

    [+ hit_ewma[0..n-1], + tanh(misses/5)[0..n-1], + t_frac ]

t_frac = step / total_steps. Output column j = activation of band j.
If the artifact is missing or unreadable the scheduler degrades gracefully to
an epsilon-greedy floor scheduler so the API never crashes.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

from .baseline import BaseScheduler


# --------------------------------------------------------------------------- #
#  the input feature vector the model sees (fixed shape = 2*n_bands + 1)
# --------------------------------------------------------------------------- #
def build_features(hits: np.ndarray, misses: np.ndarray, t: int, n_steps: int) -> np.ndarray:
    """hits   = per-band EWMA hit evidence (0..1ish)
       misses = per-band miss counts
    """
    return np.concatenate(
        [
            np.clip(np.asarray(hits, dtype=float), 0.0, 1.0),
            np.tanh(np.asarray(misses, dtype=float) / 5.0),
            [float(t) / max(1, int(n_steps))],
        ]
    )


# --------------------------------------------------------------------------- #
#  loading any pickle / joblib artifact that satisfies the contract
# --------------------------------------------------------------------------- #
def load_model(path: str | Path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            pass
    try:
        import joblib  # type: ignore

        return joblib.load(p)
    except Exception:
        warnings.warn(f"dataset_rfi: could not load artifact {p} (falling back to ε-greedy floor)")
        return None


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """Return (1, C) float scores from ANY contract-compliant model."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
    else:
        p = np.asarray(model.predict(X), dtype=float)
    return p.reshape(1, -1)


# --------------------------------------------------------------------------- #
#  the scheduler
# --------------------------------------------------------------------------- #
class DatasetRFIScheduler(BaseScheduler):
    name = "dataset_rfi"

    def __init__(
        self,
        n_bands: int,
        artifact: str | Path | None = None,
        seed: int = 0,
        alpha: float = 0.25,
        epsilon: float = 0.05,
        floor: float = 0.15,
        n_steps: int = 300,
    ):
        super().__init__(n_bands, seed)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.floor = float(floor)
        self.n_steps = int(n_steps)
        self.model = load_model(artifact) if artifact else None
        self.hits = np.zeros(self.n, dtype=float)
        self.misses = np.zeros(self.n, dtype=float)
        self.q = np.zeros(self.n, dtype=float)  # belief heatmap (dashboards)
        self._sweep = 0

    def tick(self, t: int) -> None:
        super().tick(t)
        self.hits *= 1.0 - self.alpha
        self.q = self.q * (1.0 - self.alpha)

    def select(self, t: int) -> int:
        # guaranteed broad coverage so surprise emitters stay findable
        if self.rng.random() < self.floor:
            band = self._sweep % self.n
            self._sweep += 1
            return band

        if self.model is None:
            # graceful degradation: ε-greedy over hit evidence
            if self.rng.random() < self.epsilon:
                band = int(self.rng.integers(self.n))
            else:
                band = int(np.argmax(self.hits))
            return band

        X = build_features(self.hits, self.misses, t, self.n_steps).reshape(1, -1)
        try:
            scores = predict_scores(self.model, X)[0]
            return int(np.argmax(scores[: self.n]))
        except Exception:  # model misbehaved -> fall back
            return int(np.argmax(self.hits))

    def update(self, band: int, hit: bool, t: int) -> None:
        if hit:
            self.hits[band] += self.alpha
            self.q[band] += self.alpha
        else:
            self.misses[band] += 1.0