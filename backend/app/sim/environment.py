from __future__ import annotations

import numpy as np

from .emitter import Emitter


class RFEnvironment:
    """Ground-truth RF activity over an N-band x T-step grid.

    The environment knows which band is transmitting at every time step —
    this is the 'truth' the receiver & scheduler never see directly, used
    only to score interception (and by the simulator for training labels).
    """

    def __init__(
        self,
        n_bands: int = 10,
        n_steps: int = 300,
        emitters: list[Emitter] | None = None,
        noise: float = 0.0,
        seed: int = 42,
        freq_start_ghz: float = 2.0,
        freq_end_ghz: float = 18.0,
    ):
        self.n_bands = int(n_bands)
        self.n_steps = int(n_steps)
        self.emitters = emitters or []
        self.noise = float(noise)
        self.seed = int(seed)
        self.freq_start_ghz = float(freq_start_ghz)
        self.freq_end_ghz = float(freq_end_ghz)
        self.rng = np.random.default_rng(self.seed)
        self.ground_truth: np.ndarray = self._build_truth()
        self.events: list[dict] = self._detect_events()

    # --------------------------------------------------------------- frequency
    def band_edges_ghz(self) -> list[list[float]]:
        """Real RF sub-range of each band (sender side operates inside here)."""
        span = self.freq_end_ghz - self.freq_start_ghz
        w = span / self.n_bands
        return [
            [round(self.freq_start_ghz + i * w, 2), round(self.freq_start_ghz + (i + 1) * w, 2)]
            for i in range(self.n_bands)
        ]

    def band_center_ghz(self, band: int) -> float:
        lo, hi = self.band_edges_ghz()[int(band)]
        return round((lo + hi) / 2, 2)

    # ------------------------------------------------------------------ truth
    def _build_truth(self) -> np.ndarray:
        truth = np.zeros((self.n_steps, self.n_bands), dtype=bool)
        for i, em in enumerate(self.emitters):
            em.rng_seed = self.seed + i * 7919
            for t in range(self.n_steps):
                band = em.active_band_at(t)
                if band is not None and 0 <= band < self.n_bands:
                    truth[t, band] = True
        if self.noise > 0:
            mask = self.rng.random((self.n_steps, self.n_bands)) < self.noise
            truth |= mask
        return truth

    def truth_at(self, t: int) -> np.ndarray:
        return self.ground_truth[t]

    # ---------------------------------------------------------------- events
    def _detect_events(self) -> list[dict]:
        """Derive salient events from ground-truth transitions."""
        events: list[dict] = []
        prev = set(np.where(self.ground_truth[0])[0].tolist())
        known_active: set[int] = set(prev)
        for t in range(1, self.n_steps):
            cur = set(np.where(self.ground_truth[t])[0].tolist())
            on = cur - prev
            off = prev - cur
            surprise = [b for b in sorted(on) if b not in known_active]
            if on or off:
                events.append(
                    {
                        "t": int(t),
                        "type": "surprise" if surprise else "change",
                        "bands_on": sorted(on),
                        "bands_off": sorted(off),
                        "surprise": sorted(surprise),
                    }
                )
            known_active |= cur
            prev = cur
        return events

    def active_segments(self) -> list[dict]:
        """Run-length decode each band into active (transmission) segments."""
        segments: list[dict] = []
        for b in range(self.n_bands):
            col = self.ground_truth[:, b]
            start = None
            for t in range(self.n_steps):
                on = bool(col[t])
                if on and start is None:
                    start = t
                if (not on or t == self.n_steps - 1) and start is not None:
                    end = t if not on else t
                    segments.append({"band": b, "start": start, "end": end})
                    start = None
        return segments

    def total_transmissions(self) -> int:
        return int(self.ground_truth.sum())

    def activity_profile(self) -> list[int]:
        return [int(self.ground_truth[:, b].sum()) for b in range(self.n_bands)]

    def as_json_grid(self) -> list[list[bool]]:
        return self.ground_truth.tolist()