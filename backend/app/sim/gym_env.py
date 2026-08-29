from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .schedulers.rl_policy import build_obs
from .environment import RFEnvironment
from .scenarios import get_emitters

EPISODE_STEPS = 200

# Scenario mix used to train the RL policy for generalisation.
TRAIN_PALETTE = [
    "stable",
    "switch",
    "surprise",
    "hopper",
    "burst",
    "multi_sim",
    "periodic",
    "random",
]


class EWGymEnv(gym.Env):
    """Gymnasium wrapper around the RF simulation for SB3 DQN/PPO training.

    Observation: [EWMA activity (n), normalised counts (n), normalised time (1)]
    Action: which band to scan next. Reward: +1 hit, -0.5 miss.
    Each episode draws a random scenario + seed from the training palette so the
    policy learns to adapt rather than memorise one pattern.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_bands: int = 10, episode_steps: int = EPISODE_STEPS, seed: int = 0):
        super().__init__()
        self.n_bands = int(n_bands)
        self.episode_steps = int(episode_steps)
        self.action_space = spaces.Discrete(self.n_bands)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * self.n_bands + 1,), dtype=np.float32
        )
        self.seed_rng = None
        self._seed = int(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(self._seed) if self._seed else np.random.default_rng()
        if seed is not None:
            rng = np.random.default_rng(seed)
        self._seed += 1
        scenario_id = rng.choice(TRAIN_PALETTE)
        self._scenario_id = scenario_id
        ems = get_emitters(scenario_id, self.n_bands, self.episode_steps)
        self._env = RFEnvironment(
            n_bands=self.n_bands,
            n_steps=self.episode_steps,
            emitters=ems,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        self.t = 0
        self.q = np.zeros(self.n_bands, dtype=float)
        self.counts = np.zeros(self.n_bands, dtype=float)
        self.cum_reward = 0.0
        self.hits = 0
        obs = build_obs(self.q, self.counts, self.t, self.n_bands).astype(np.float32)
        return obs, {"scenario": scenario_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        truth = bool(self._env.ground_truth[self.t, action])
        reward = 1.0 if truth else -0.5
        self.counts[action] += 1.0
        if truth:
            self.q[action] += 0.25
            self.hits += 1
        self.q = self.q * 0.75
        self.cum_reward += reward
        self.t += 1
        obs = build_obs(self.q, self.counts, self.t, self.n_bands).astype(np.float32)
        truncated = self.t >= self.episode_steps
        info = {
            "hit": bool(truth),
            "cum_reward": float(self.cum_reward),
            "n_transmissions": int(self._env.ground_truth.sum()),
            "hits": self.hits,
            "scenario": getattr(self, "_scenario_id", ""),
        }
        return obs, float(reward), False, truncated, info

    def episode_interception(self) -> float:
        """Cell-level hit fraction for the current episode (eval curve)."""
        total = int(self._env.ground_truth.sum())
        return float(self.hits) / total if total else 0.0