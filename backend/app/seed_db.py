"""Populate database/spectra.db with representative sample runs.

Usage:  python -m app.seed_db   (from backend/)

Records ~6 varied runs across schedulers/scenarios so the dashboard's
"Recorded runs" panel and /api/db/stats show data immediately.
"""

from __future__ import annotations

from app import db
from app.config import SimConfig
from app.sim.engine import run_simulation

SAMPLES = [
    dict(scenario="stable_switch_surprise", scheduler="thompson", n_steps=200, seed=2024),
    dict(scenario="switch", scheduler="ucb1", n_steps=200, seed=11),
    dict(scenario="surprise", scheduler="epsilon_greedy", n_steps=160, seed=5),
    dict(scenario="periodic_only", scheduler="sequence", n_steps=200, seed=42),
    dict(scenario="hopper", scheduler="sequence", n_steps=200, seed=7),
    dict(scenario="stable", scheduler="rl_ppo", n_steps=120, seed=9),
    dict(scenario="hopper", scheduler="rl_dqn", n_steps=200, seed=3),
    dict(scenario="threat", scheduler="thompson", n_steps=200, seed=13),
]


def seed() -> None:
    print(f"db: {db.init()}")
    for s in SAMPLES:
        cfg = SimConfig(n_bands=10, n_steps=s["n_steps"], seed=s["seed"], scenario=s["scenario"], scheduler=s["scheduler"])
        res = run_simulation(cfg)
        rid = db.save_run(res)
        sm, bm = res["smart"]["metrics"], res["baseline"]["metrics"]
        print(
            f"  + {s['scheduler']:14s} {s['scenario']:22s} "
            f"smart ir {sm['interception_ratio']:5.1f} % vs base {bm['interception_ratio']:5.1f} % "
            f" | reward {sm['avg_reward']:+.3f} | {rid[:8]}"
        )
    print(f"total runs in db: {db.stats()['total_runs']}")


if __name__ == "__main__":
    seed()