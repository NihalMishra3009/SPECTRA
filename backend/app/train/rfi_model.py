"""Demo importable model class for the dataset_rfi contract.

Kept OUTSIDE any __main__-runable script so pickle stores the stable module
path (``app.train.rfi_model``) and the artifact loads everywhere — including
from the live FastAPI server.
"""

from __future__ import annotations

import numpy as np


class BandPredictor:
    """Next-active-band softmax classifier — pure NumPy, no sklearn needed.

    Contract-compliant: ``predict_proba(X)`` with X shape (N, 2*n_bands+1)
    and output (N, C) per-band activation scores.
    """

    def __init__(self, W: np.ndarray, n_bands: int):
        self.W = np.asarray(W, dtype=float)
        self.n_bands = int(n_bands)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.W.T
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


def softmax_fit(X: np.ndarray, y: np.ndarray, n_classes: int, iters: int = 250, lr: float = 0.25) -> np.ndarray:
    """Gradient-descent multinomial logistic regression → weight matrix."""
    W = np.zeros((n_classes, X.shape[1]), dtype=float)
    Y = np.zeros((len(y), n_classes), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    for _ in range(iters):
        z = X @ W.T
        p = np.exp(z - z.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        W += lr * ((Y - p).T @ X) / len(X)
    return W