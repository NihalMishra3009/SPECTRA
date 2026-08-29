from __future__ import annotations

import numpy as np

from .environment import RFEnvironment


class Receiver:
    """Bandwidth-limited scan receiver: sees only the band it is tuned to.

    The receiver physically cannot see the rest of the spectrum — that is
    the core constraint the scheduler has to work around, exactly like a
    superheterodyne swept receiver in a real EW suite.
    """

    def __init__(self, env: RFEnvironment, snr_noise: float = 0.0):
        self.env = env
        self.snr_noise = snr_noise
        self.rng = np.random.default_rng(env.seed + 1337)

    def observe(self, band: int, t: int) -> tuple[bool, float]:
        """Return (hit, snr) for scanning `band` at step `t`."""
        truth = bool(self.env.ground_truth[t, band])
        if not truth:
            # Pfa modelling: occasionally a noise spike registers as a hit
            if self.snr_noise > 0 and self.rng.random() < self.snr_noise:
                return True, float(self.rng.uniform(0.1, 0.5))
            return False, 0.0
        snr = 1.0 + float(self.rng.uniform(0.0, 1.0))
        return True, snr