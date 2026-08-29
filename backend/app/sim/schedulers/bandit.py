from __future__ import annotations

import numpy as np

from .baseline import BaseScheduler


class _RecencyMixin(BaseScheduler):
    """Shared machinery: per-step EWMA decay + per-band recency updates.

    Every step every band's estimate decays by (1 - alpha); a hit raises the
    band back up. Old evidence therefore fades (adaptation), while recently
    active bands stay hot. Exploration floors keep surprise emitters findable.

    `floor` = guaranteed fraction of scans spent on a rotating broad sweep
    (uniform coverage) so nobody can starve the scheduler's long tail. This
    mirrors real EW practice: fast wide sweep + dwell on probable emitters.
    """

    def __init__(self, n_bands: int, alpha: float, floor: float = 0.15, seed: int = 0):
        super().__init__(n_bands, seed)
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.q = np.zeros(self.n, dtype=float)
        self.counts = np.zeros(self.n, dtype=float)
        self._sweep = 0

    def _maybe_floor(self) -> int | None:
        if self.rng.random() < self.floor:
            band = self._sweep % self.n
            self._sweep += 1
            return band
        return None

    def tick(self, t: int) -> None:
        super().tick(t)
        self.q = self.q * (1.0 - self.alpha)

    def update(self, band: int, hit: bool, t: int) -> None:
        self.counts[band] += 1.0
        if hit:
            self.q[band] += self.alpha  # reward raises the band's estimate

    def _first_untried(self) -> int | None:
        for b in range(self.n):
            if self.counts[b] < 1.0:
                return b
        return None


class EpsilonGreedy(_RecencyMixin):
    name = "epsilon_greedy"

    def __init__(self, n_bands: int, alpha: float, epsilon: float, floor: float = 0.0, seed: int = 0):
        super().__init__(n_bands, alpha, floor, seed)
        self.epsilon = float(epsilon)

    def select(self, t: int) -> int:
        u = self._first_untried()
        if u is not None:
            return u
        x = self.rng.random()
        if x < self.epsilon:
            return int(self.rng.integers(self.n))
        if x < self.epsilon + self.floor:
            return self._sweep_step()
        return int(np.argmax(self.q))

    def _sweep_step(self) -> int:
        band = self._sweep % self.n
        self._sweep += 1
        return band


class UCB1(_RecencyMixin):
    name = "ucb1"

    def __init__(self, n_bands: int, alpha: float, c: float = 1.2, floor: float = 0.15, seed: int = 0):
        super().__init__(n_bands, alpha, floor, seed)
        self.c = float(c)

    def select(self, t: int) -> int:
        u = self._first_untried()
        if u is not None:
            return u
        floor_band = self._maybe_floor()
        if floor_band is not None:
            return floor_band
        total = float(self.t) + 1.0
        conf = self.c * np.sqrt(np.log(total + 2.0) / (self.counts + 1.0))
        ucb = self.q + conf
        return int(np.argmax(ucb))


class ThompsonSampling(_RecencyMixin):
    """Posterior sampling over a Gaussian belief per band."""

    name = "thompson"

    def __init__(self, n_bands: int, alpha: float, floor: float = 0.15, seed: int = 0):
        super().__init__(n_bands, alpha, floor, seed)

    def select(self, t: int) -> int:
        u = self._first_untried()
        if u is not None:
            return u
        floor_band = self._maybe_floor()
        if floor_band is not None:
            return floor_band
        sigma = 1.0 / np.sqrt(self.counts + 1.0)
        samples = self.rng.normal(self.q, sigma)
        return int(np.argmax(samples))