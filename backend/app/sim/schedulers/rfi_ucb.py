"""Friend's dataset-trained RFI model integrated as a UCB1 blend.

Model: RandomForestClassifier trained on the TSRD (Turing Synthetic Radar
Dataset) — inputs 4 per-band features:
    [prev_hit, recent_hit_rate, windows_since_hit, avg_amplitude]
outputs P(this band is active in the NEXT time window).

The scheduler feeds that probability in as a ```prior``` on top of a UCB1
bandit score (the same recipe the model's own README proves: 49% vs 7% for
round-robin). Every feature is tracked live inside the receiver loop — no
external windowing needed.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .baseline import BaseScheduler
from .dataset_rfi import load_model

FEATURES = ["prev_hit", "recent_hit_rate", "windows_since_hit", "avg_amplitude"]


def _model_active_probas(model, X: np.ndarray) -> np.ndarray:
    """Call any sklearn-ish model on (n, 4) rows -> P(active next window) per band.

    The RF was fitted on a named DataFrame, so we rebuild those column names to
    keep predictions byte-compatible with the trainer (and silence sklearn).
    """
    X = np.asarray(X, dtype=float).reshape(-1, 4)
    try:
        import pandas as pd

        X = pd.DataFrame(X, columns=list(FEATURES))
    except ImportError:
        pass
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X), dtype=float)
        return p[:, 1] if p.ndim == 2 and p.shape[1] >= 2 else np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


class RFIUCBScheduler(BaseScheduler):
    """UCB1 + ML activity prior (the friend's model) + sweep coverage floor."""

    name = "rfi_ucb"

    def __init__(
        self,
        n_bands: int,
        artifact: str | Path | None = None,
        seed: int = 0,
        alpha: float = 0.25,
        c: float = 1.2,
        floor: float = 0.15,
        blend: float = 0.5,
        windows: int = 5,
    ):
        super().__init__(n_bands, seed)
        self.alpha = float(alpha)
        self.c = float(c)
        self.floor = float(floor)
        self.blend = float(blend)
        self.windows = int(windows)
        self.q = np.zeros(self.n, dtype=float)
        self.counts = np.zeros(self.n, dtype=float)
        # live per-band trackers (bounded ring buffers)
        self.hits = [np.zeros(self.windows, dtype=bool) for _ in range(self.n)]
        self.hit_pos = np.zeros(self.n, dtype=int)
        self.since_hit = np.zeros(self.n, dtype=float)  # steps since last hit
        self.model = load_model(artifact) if artifact else None
        self.model_name = Path(artifact).name if artifact and Path(artifact).exists() else None
        self._sweep = 0
        self._calls = 0
        self._latency_ms = 0.0
        self.last_scores: list[float] = []

    # ------------------------------------------------------------- features
    def _features(self) -> np.ndarray:
        """(n, 4) feature matrix in model order for all bands at step t."""
        rows = []
        for b in range(self.n):
            h = self.hits[b]
            any_seen = self.hit_pos[b] > 0
            prev_hit = 1.0 if any_seen and h[(self.hit_pos[b] - 1) % self.windows] else 0.0
            rate = float(h.sum()) / self.windows if any_seen else 0.0
            since = min(self.since_hit[b], self.windows * 10) / float(self.windows * 10)
            amp = 1.0 if any_seen and h.mean() > 0 else 0.0  # snr proxy
            rows.append([prev_hit, rate, since, amp])
        return np.array(rows, dtype=float)

    # ------------------------------------------------------------- scheduler
    def tick(self, t: int) -> None:
        super().tick(t)
        self.q = self.q * (1.0 - self.alpha)

    def select(self, t: int) -> int:
        if self.rng.random() < self.floor:
            band = self._sweep % self.n
            self._sweep += 1
            return band

        total = float(self.t) + 1.0
        ucb = self.q + self.c * np.sqrt(np.log(total + 2.0) / (self.counts + 1.0))

        if self.model is None:
            if self.rng.random() < 0.05:
                return int(self.rng.integers(self.n))
            return int(np.argmax(ucb))

        try:
            t0 = time.perf_counter()
            prior = _model_active_probas(self.model, self._features())
            self._latency_ms += (time.perf_counter() - t0) * 1000.0
            self._calls += 1
            self.last_scores = [round(float(x), 4) for x in prior]
            score = ucb + self.blend * prior  # ML prior blended into the bandit
            return int(np.argmax(score))
        except Exception:
            return int(np.argmax(ucb))

    def update(self, band: int, hit: bool, t: int) -> None:
        self.counts[band] += 1.0
        self.hits[band][self.hit_pos[band] % self.windows] = bool(hit)
        self.hit_pos[band] += 1
        if hit:
            self.q[band] += self.alpha
            self.since_hit[band] = 0.0
        else:
            self.since_hit[band] += 1.0

    # ------------------------------------------------------------------ status
    def describe(self) -> dict:
        """Live model telemetry — surfaced in the response/dashboard as proof
        that the external model is actually running and answering every step."""
        info = {
            "name": self.name,
            "scheduler": self.__class__.__name__,
            "model_file": self.model_name,
            "loaded": self.model is not None,
            "features": FEATURES,
            "predict_calls": self._calls,
            "avg_latency_ms": round(self._latency_ms / max(1, self._calls), 3),
            "last_scores": self.last_scores,
        }
        if self.model is not None:
            m = self.model
            info["model_type"] = type(m).__name__
            info["trees"] = getattr(m, "n_estimators", None)
            info["max_depth"] = getattr(m, "max_depth", None)
            info["n_features"] = getattr(m, "n_features_in_", None)
        return info