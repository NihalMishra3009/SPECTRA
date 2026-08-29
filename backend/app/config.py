from __future__ import annotations

from pydantic import BaseModel, Field


class SimConfig(BaseModel):
    """Runtime configuration for a single simulation run."""

    n_bands: int = Field(default=10, ge=2, le=64, description="Number of frequency bands")
    n_steps: int = Field(default=300, ge=10, le=5000, description="Number of time steps")
    seed: int = Field(default=42, ge=0, description="Random seed (deterministic replay)")
    scenario: str = Field(default="stable_switch_surprise", description="Scenario preset id")
    scheduler: str = Field(default="ucb1", description="Scheduler for the smart scanner")
    alpha: float = Field(default=0.25, gt=0.0, le=1.0, description="Recency decay / learning rate")
    epsilon: float = Field(default=0.05, ge=0.0, le=1.0, description="Exploration probability (epsilon-greedy)")
    window: int = Field(default=40, ge=1, le=500, description="Sliding-window size (adaptive flavor)")
    floor: float = Field(default=0.15, ge=0.0, le=1.0, description="Guaranteed broad-sweep coverage fraction")
    freq_start_ghz: float = Field(default=2.0, ge=0.0, description="Low edge of guarded spectrum (GHz)")
    freq_end_ghz: float = Field(default=18.0, description="High edge of guarded spectrum (GHz)")


SCHEDULERS = {
    "round_robin": "Fixed open-loop sweep (baseline anchor)",
    "epsilon_greedy": "Epsilon-greedy bandit with recency decay",
    "ucb1": "UCB1 bandit with recency decay",
    "thompson": "Thompson Sampling bandit with recency decay",
    "adaptive_window": "Sliding-window recency scheduler",
    "rl_dqn": "Trained DQN policy (Stable-Baselines3)",
    "rl_ppo": "Trained PPO policy (Stable-Baselines3)",
    "sequence": "Sequence-aware LSTM timing/hop predictor",
}

DEFAULT_DEMO = {
    "n_bands": 10,
    "n_steps": 300,
    "seed": 2024,
    "scenario": "stable_switch_surprise",
    "scheduler": "ucb1",
    "alpha": 0.25,
    "epsilon": 0.05,
    "window": 40,
    "floor": 0.15,
}