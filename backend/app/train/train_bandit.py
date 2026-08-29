"""Evaluate the online bandits across the 12-scenario catalog for comparison.

Writes artifacts/bandit_baseline.json (used by the dashboard Model panel) and
prints a markdown table.

Usage:
    python -m app.train.train_bandit
"""
from __future__ import annotations

import argparse
import json

from ..config import SimConfig
from ..sim import engine
from ..sim.scenarios import scenario_catalog
from . import ARTIFACTS_DIR

SCHEDULERS = ["epsilon_greedy", "ucb1", "thompson", "adaptive_window"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bands", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for sched in SCHEDULERS:
        summed = 0.0
        n = 0
        per: dict[str, float] = {}
        for sc in scenario_catalog():
            cfg = SimConfig(
                n_bands=args.bands,
                n_steps=args.steps,
                seed=args.seed,
                scenario=sc["id"],
                scheduler=sched,
            )
            res = engine.run_simulation(cfg)
            ir = res["smart"]["metrics"]["interception_ratio"]
            per[sc["id"]] = ir
            summed += ir
            n += 1
        summary[sched] = {
            "mean_interception_pct": round(summed / max(1, n), 2),
            "per_scenario": per,
        }
        rows.append(
            f"| {sched} | {round(summed / max(1, n), 2)}% |"
        )

    (ARTIFACTS_DIR / "bandit_baseline.json").write_text(
        json.dumps({"schedulers": SCHEDULERS, "summary": summary, "seed": args.seed}, indent=2)
    )

    print("| Scheduler | Mean interception (%) |")
    print("|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()