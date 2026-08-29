from __future__ import annotations

import numpy as np

from .baseline import BaseScheduler


class SlidingWindow(BaseScheduler):
    """Windowed recency weighting: activity = mean of last `window` scans.

    Used by the 'adaptive' flavor to demonstrate pattern-switch response —
    the scheduler re-prioritizes as soon as the stale window rolls off.
    """

    name = "adaptive_window"

    def __init__(self, n_bands: int, window: int, seed: int = 0, epsilon: float = 0.05):
        super().__init__(n_bands, seed)
        self.window = int(window)
        self.epsilon = float(epsilon)
        self.history: list[tuple[int, bool]] = []

    def select(self, t: int) -> int:
        recent: dict[int, int] = {}
        for band, hit in self.history[-self.window :]:
            if hit:
                recent[band] = recent.get(band, 0) + 1
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n))
        if not recent:
            return int(self.rng.integers(self.n))
        best = max(recent, key=lambda b: (recent[b], b))
        return int(best)

    def update(self, band: int, hit: bool, t: int) -> None:
        self.history.append((band, hit))
        if len(self.history) > 5000:
            self.history = self.history[-2000:]