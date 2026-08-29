from __future__ import annotations

from pathlib import Path

import numpy as np

from .baseline import BaseScheduler


def build_obs(q: np.ndarray, counts: np.ndarray, t: int, n_bands: int) -> np.ndarray:
    """Observation vector shared by RL training (gym env) and inference.

    layout = [EWMA activity per band (n), normalised counts (n), normalised time (1)]
    Unit mismatch between train (200-step episodes) and demo (300 steps) is absorbed
    by the fixed normalisation constant below.
    """
    q_c = np.clip(np.asarray(q, dtype=float), 0.0, 1.0).reshape(n_bands)
    maxc = float(counts.max()) if counts.size else 0.0
    c_norm = np.clip(np.asarray(counts, dtype=float) / max(maxc, 1e-9), 0.0, 1.0)
    t_norm = np.asarray([min(t / 500.0, 1.0)], dtype=float)
    return np.concatenate([q_c, c_norm, t_norm])


class RlPolicyScheduler(BaseScheduler):
    """Serve a trained Stable-Baselines3 policy as the live scheduler.

    Same EMWA state estimator the bandits use; the decision is the policy's.
    """

    name = "rl_policy"

    def __init__(self, n_bands: int, artifact: str | Path, seed: int = 0, alpha: float = 0.25):
        super().__init__(n_bands, seed)
        self.alpha = float(alpha)
        self.q = np.zeros(self.n, dtype=float)
        self.counts = np.zeros(self.n, dtype=float)
        self._art = str(artifact)
        try:
            from stable_baselines3 import DQN, PPO  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("stable-baselines3 required for RL policy") from e
        self.model = self._load(self._art)
        self.algo = self._detect_algo(self.model)

    @staticmethod
    def _load(path: str):
        from stable_baselines3 import DQN, PPO

        for cls in (DQN, PPO):
            try:
                return cls.load(path)
            except Exception:
                continue
        raise RuntimeError(f"could not load SB3 artifact: {path}")

    @staticmethod
    def _detect_algo(model) -> str:
        name = type(model).__name__.lower()
        return "dqn" if "dqn" in name else "ppo" if "ppo" in name else "rl"

    def tick(self, t: int) -> None:
        super().tick(t)
        self.q = self.q * (1.0 - self.alpha)

    def update(self, band: int, hit: bool, t: int) -> None:
        self.counts[band] += 1.0
        if hit:
            self.q[band] += self.alpha

    def select(self, t: int) -> int:
        obs = build_obs(self.q, self.counts, t, self.n).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    def state(self) -> dict:
        s = super().state()
        s.update({"artifact": self._art, "algo": self.algo})
        return s