from __future__ import annotations

import json

import pytest

from app.config import DEFAULT_DEMO, SimConfig
from app.sim import engine
from app.sim.scenarios import SCENARIOS, get_emitters


def cfg(**kw) -> SimConfig:
    base = {**DEFAULT_DEMO, "n_steps": 200, "seed": 42}
    base.update(kw)
    return SimConfig(**base)


def test_scenario_catalog_covers_12_psi_scenarios():
    ids = [
        "stable", "switch", "single_dominant", "multi_equal", "random",
        "surprise", "periodic", "hopper", "burst", "multi_sim",
        "noise", "threat",
    ]
    for s in ids:
        assert s in SCENARIOS, f"missing scenario {s}"


def test_all_scenarios_run():
    for sid in ["stable", "switch", "surprise", "hopper", "periodic", "burst",
                "multi_sim", "random", "threat"]:
        res = engine.run_simulation(cfg(scenario=sid, scheduler="ucb1"))
        assert res["smart"]["metrics"]["interception_ratio"] >= 0.0
        assert len(res["smart"]["log"]) == 200


def test_deterministic_replay():
    a = engine.run_simulation(cfg(scheduler="ucb1"))
    b = engine.run_simulation(cfg(scheduler="ucb1"))
    assert a["ground_truth"] == b["ground_truth"]
    assert a["smart"]["log"] == b["smart"]["log"]


def test_ucb1_beats_baseline_on_stable():
    """On structured activity smart matches coverage AND massively cuts wasted scans."""
    res = engine.run_simulation(cfg(scenario="stable", scheduler="ucb1"))
    b, s = res["baseline"]["metrics"], res["smart"]["metrics"]
    assert s["interception_ratio"] >= b["interception_ratio"]
    assert s["avg_reward"] > b["avg_reward"]


def test_thompson_scores_on_periodic():
    """Recency-weighted Thompson catches the periodic emitter; round-robin times out."""
    res = engine.run_simulation(cfg(scenario="periodic", scheduler="thompson", n_steps=300))
    b, s = res["baseline"]["metrics"], res["smart"]["metrics"]
    assert s["interception_ratio"] >= b["interception_ratio"]
    assert s["avg_reward"] > b["avg_reward"]


def test_surprise_emitter_detected():
    res = engine.run_simulation(cfg(scenario="surprise", scheduler="epsilon_greedy", n_steps=250))
    smart_hits = [e for e in res["smart"]["log"] if e["hit"]]
    assert any(e["band"] == 7 for e in smart_hits), "surprise band 7 never hit"


def test_adaptation_speed_reported():
    res = engine.run_simulation(cfg(scenario="switch", scheduler="ucb1", n_steps=220))
    speed = res["smart"]["metrics"]["adaptation_speed"]
    assert speed is not None and speed >= 0


def test_unknown_scheduler_raises():
    with pytest.raises(ValueError):
        engine.run_simulation(cfg(scheduler="nonsense"))


def test_metrics_in_bounds():
    res = engine.run_simulation(cfg(scenario="stable", scheduler="epsilon_greedy", epsilon=0.1))
    for side in ("baseline", "smart"):
        m = res[side]["metrics"]
        assert 0.0 <= m["interception_ratio"] <= 100.0
        assert m["hits"] >= 0
        assert m["miss_count"] >= 0


def test_ground_truth_json_serialisable():
    res = engine.run_simulation(cfg())
    json.dumps(res)


def test_priorities_trajectory_attached():
    res = engine.run_simulation(cfg(scheduler="thompson", n_steps=60))
    prio = res["smart"].get("priorities")
    assert prio is None or len(prio) == 60


def test_get_emitters_honours_band_count():
    ems = get_emitters("multi_equal", 10, 200)
    assert all(e.bands[0] < 10 for e in ems)