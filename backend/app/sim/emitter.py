from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Emitter:
    """A single emitter (transmitter) in the simulated RF environment.

    - Fixed emitters: interval=1 (always on inside their time window).
    - Periodic emitters: interval>1 (on every `interval` steps starting at `phase`).
    - Hopping/agile: `hop_bands` provided — picks one of them per on-pulse.
    - Bursty/intermittent: `duty` < 1 — active with probability `duty` on online steps.
    """

    name: str
    bands: list[int] = field(default_factory=list)
    interval: int = 1
    phase: int = 0
    start: int = 0
    end: int | None = None
    hop_bands: list[int] | None = None
    burst: float | None = None  # duty cycle in (0,1] if intermittent
    threat_weight: float = 1.0  # priority bonus (Scenario 12)
    rng_seed: int = 0

    def _w(self) -> int:
        return self.rng_seed

    def active_band_at(self, t: int) -> int | None:
        if t < self.start:
            return None
        if self.end is not None and t > self.end:
            return None
        if self.burst is not None:
            # deterministic pseudo-burstiness from seed so runs are reproducible
            import hashlib

            h = int(hashlib.md5(f"{self.name}:{t}:{self.rng_seed}".encode()).hexdigest(), 16)
            if h % 1000 >= int(self.burst * 1000):
                return None
        if self.interval <= 1:
            band = self.bands[0] if self.bands else 0
        else:
            if t % self.interval != self.phase:
                return None
            if self.hop_bands:
                idx = (t // self.interval) % len(self.hop_bands)
                band = self.hop_bands[idx]
            else:
                band = self.bands[0] if self.bands else 0
        # probability that this emitter actually transmits given burst duty
        return band

    def is_active_at(self, t: int) -> bool:
        return self.active_band_at(t) is not None