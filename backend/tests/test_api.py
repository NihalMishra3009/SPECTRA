"""End-to-end API tests — every REST endpoint + the WebSocket stream."""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.config import DEFAULT_DEMO, SCHEDULERS
from app.main import app
from app.sim.scenarios import scenario_catalog

client = TestClient(app)


# --------------------------------------------------------------------------- #
#  GET /api/health
# --------------------------------------------------------------------------- #
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --------------------------------------------------------------------------- #
#  GET /api/scenarios
# --------------------------------------------------------------------------- #
def test_scenarios():
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    scen = r.json()["scenarios"]
    assert len(scen) == 14
    for s in scen:
        assert {"id", "label", "desc"} <= set(s)


def test_scenario_ids_usable_by_simulate():
    scen = client.get("/api/scenarios").json()["scenarios"]
    for s in scen:
        r = client.post("/api/simulate", json={"scenario": s["id"], "n_steps": 30, "scheduler": "ucb1"})
        assert r.status_code == 200, f"scenario {s['id']} failed: {r.text}"


# --------------------------------------------------------------------------- #
#  GET /api/schedulers
# --------------------------------------------------------------------------- #
def test_schedulers():
    r = client.get("/api/schedulers")
    assert r.status_code == 200
    scheds = r.json()["schedulers"]
    assert len(scheds) == 10
    assert set(SCHEDULERS) == {s["id"] for s in scheds}
    assert r.json()["defaults"] == DEFAULT_DEMO


def test_scheduler_ids_usable_by_simulate():
    scheds = client.get("/api/schedulers").json()["schedulers"]
    for s in scheds:
        r = client.post("/api/simulate", json={"scenario": "stable", "n_steps": 40, "scheduler": s["id"]})
        assert r.status_code == 200, f"scheduler {s['id']} failed: {r.text}"
        assert "metrics" in r.json()["smart"]


# --------------------------------------------------------------------------- #
#  GET /api/models
# --------------------------------------------------------------------------- #
def test_models_all_present():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    ids = {m["scheduler"]: m for m in models}
    assert set(ids) == {"rl_dqn", "rl_ppo", "bandit_baseline", "sequence", "dataset_rfi", "rfi_ucb"}
    for m in models:
        assert m["present"] is True, f"{m['scheduler']} artifact missing"


# --------------------------------------------------------------------------- #
#  GET /api/curves/<file>
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("file", ["dqn_curves.json", "ppo_curves.json", "sequence_curves.json"])
def test_curves_serve(file):
    r = client.get(f"/api/curves/{file}")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) >= 1
    assert isinstance(data[0], dict)


def test_curves_missing_404():
    r = client.get("/api/curves/does_not_exist.json")
    assert r.status_code == 404
    assert r.json()["error"] == "curve not found"


# --------------------------------------------------------------------------- #
#  POST /api/simulate — full result structure
# --------------------------------------------------------------------------- #
def test_simulate_default_structure():
    r = client.post("/api/simulate", json={})
    assert r.status_code == 200
    j = r.json()
    for key in (
        "config", "scenario_id", "scenario_label", "ground_truth", "band_edges_ghz",
        "activity_profile", "events", "n_segments", "total_transmissions",
        "baseline", "smart", "meta",
    ):
        assert key in j, f"missing {key}"
    assert len(j["ground_truth"]) == DEFAULT_DEMO["n_steps"]
    assert len(j["ground_truth"][0]) == DEFAULT_DEMO["n_bands"]
    assert len(j["band_edges_ghz"]) == DEFAULT_DEMO["n_bands"]
    for side in ("baseline", "smart"):
        s = j[side]
        assert len(s["log"]) == DEFAULT_DEMO["n_steps"]
        for k in ("interception_ratio", "avg_intercept_time", "probability_of_detection",
                  "probability_of_false_alarm", "miss_count", "adaptation_speed",
                  "avg_reward", "hits"):
            assert k in s["metrics"]


def test_simulate_deterministic_same_seed():
    body = {"scenario": "stable_switch_surprise", "scheduler": "thompson", "seed": 99, "n_steps": 80}
    a = client.post("/api/simulate", json=body).json()
    b = client.post("/api/simulate", json=body).json()
    assert a["smart"]["metrics"] == b["smart"]["metrics"]
    assert a["baseline"]["log"] == b["baseline"]["log"]
    assert a["ground_truth"] == b["ground_truth"]


# --------------------------------------------------------------------------- #
#  validation errors
# --------------------------------------------------------------------------- #
def test_simulate_bad_scenario_400():
    r = client.post("/api/simulate", json={"scenario": "not_a_real_scenario"})
    assert r.status_code == 400
    assert "unknown scenario" in r.json()["detail"]


def test_simulate_bad_scheduler_400():
    r = client.post("/api/simulate", json={"scheduler": "no_such_scheduler"})
    assert r.status_code == 400
    assert "unknown scheduler" in r.json()["detail"]


def test_simulate_bad_bounds_422():
    r = client.post("/api/simulate", json={"n_bands": -1})
    assert r.status_code == 422
    r = client.post("/api/simulate", json={"n_steps": 0})
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
#  WebSocket /ws/simulate
# --------------------------------------------------------------------------- #
def test_ws_simulate_streams_events():
    with client.websocket_connect("/ws/simulate") as ws:
        ws.send_text(json.dumps({"scenario": "stable_switch_surprise", "scheduler": "thompson", "n_steps": 30}))
        first = ws.receive_json()
        assert {"t", "truth", "baseline", "smart"} <= set(first)
        assert first["t"] == 0
        assert first["smart"]["band"] in range(10)
        frames = [first]
        for _ in range(9):
            frames.append(ws.receive_json())
        assert len(frames) == 10
        ts = [f["t"] for f in frames]
        assert ts == list(range(10))


def test_ws_bad_payload_falls_back_to_default():
    with client.websocket_connect("/ws/simulate") as ws:
        ws.send_text("not json")
        first = ws.receive_json()
        assert first["t"] == 0
        assert first["baseline"]["band"] in range(DEFAULT_DEMO["n_bands"])


# --------------------------------------------------------------------------- #
#  route table sanity
# --------------------------------------------------------------------------- #
def _all_paths(container):
    for r in container:
        if hasattr(r, "path"):
            yield r.path
        if hasattr(r, "routes"):
            yield from _all_paths(r.routes)


def test_all_routes_registered():
    # REST surface is exposed in the OpenAPI schema; WS connectivity is
    # covered directly by the /ws/simulate tests below.
    rest = set(app.openapi()["paths"].keys())
    for expect in (
        "/api/health", "/api/scenarios", "/api/schedulers", "/api/models",
        "/api/curves/{name}", "/api/simulate",
    ):
        assert expect in rest, f"route {expect} not exposed in OpenAPI"
    inner = [r for r in app.routes if type(r).__name__ == "_IncludedRouter"]
    if inner:
        ws_paths = {getattr(r, "path", None) for r in getattr(inner[0], "original_router", None).routes or []}
        assert "/ws/simulate" in ws_paths