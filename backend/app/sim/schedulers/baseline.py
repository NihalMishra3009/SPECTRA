from __future__ import annotations

import numpy as np


class BaseScheduler:
    """Common protocol: tick(t) -> select(t) -> update(band, hit, t)."""

    name = "base"

    def __init__(self, n_bands: int, seed: int = 0):
        self.n = int(n_bands)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def tick(self, t: int) -> None:
        self.t = t

    def select(self, t: int) -> int:
        raise NotImplementedError

    def update(self, band: int, hit: bool, t: int) -> None:
        raise NotImplementedError

    def state(self) -> dict:
        return {"t": self.t, "n_bands": self.n}


class RoundRobin(BaseScheduler):
    """The 'andha sweep' — traditional open-loop fixed scan. Never adapts."""

    name = "round_robin"

    def select(self, t: int) -> int:
        return int((t % self.n + self.n) % self.n)

    def update(self, band: int, hit: bool, t: int) -> None:
        pass