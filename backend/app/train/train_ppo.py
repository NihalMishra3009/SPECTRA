"""Train the PPO policy offline and save into app/train/artifacts/.

Usage:
    python -m app.train.train_ppo [--steps 60000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ..sim.gym_env import EWGymEnv, TRAIN_PALETTE
from . import ARTIFACTS_DIR


class Reporter(BaseCallback):
    def __init__(self, eval_every: int = 6000, n_eval: int = 8, seed: int = 0):
        super().__init__()
        self.eval_every = int(eval_every)
        self.n_eval = int(n_eval)
        self.seed = int(seed)
        self.log: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_every == 0:
            envs = [EWGymEnv(seed=self.seed + i * 103) for i in range(self.n_eval)]
            rews, ints = [], []
            for env in envs:
                obs, _ = env.reset()
                done, acc = False, 0.0
                while not done:
                    act, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, _ = env.step(int(act))
                    acc += float(r)
                    done = term or trunc
                rews.append(acc)
                ints.append(env.episode_interception())
            self.log.append(
                {
                    "timestep": int(self.n_calls),
                    "mean_reward": round(float(np.mean(rews)), 3),
                    "mean_interception": round(float(np.mean(ints)) * 100, 2),
                }
            )
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=6000)
    args = ap.parse_args()

    t0 = time.time()
    env = EWGymEnv(seed=args.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=args.seed,
        verbose=0,
        policy_kwargs={"net_arch": [64, 64]},
    )
    reporter = Reporter(eval_every=args.eval_every, seed=args.seed + 11)
    model.learn(total_timesteps=args.steps, callback=reporter)
    model.save(str(ARTIFACTS_DIR / "ppo.zip"))

    meta = {
        "algo": "ppo",
        "framework": "stable-baselines3",
        "obs_dim": 2 * env.n_bands + 1,
        "n_bands": env.n_bands,
        "episode_steps": env.episode_steps,
        "total_timesteps": args.steps,
        "seed": args.seed,
        "training_palette": list(TRAIN_PALETTE),
        "wall_seconds": round(time.time() - t0, 1),
        "final_interception_pct": reporter.log[-1]["mean_interception"] if reporter.log else None,
    }
    (ARTIFACTS_DIR / "ppo_meta.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACTS_DIR / "ppo_curves.json").write_text(json.dumps(reporter.log, indent=2))
    print(f"[ppo] trained in {meta['wall_seconds']}s -> artifacts/ppo.zip")
    print(json.dumps(reporter.log[-3:], indent=2))


if __name__ == "__main__":
    main()