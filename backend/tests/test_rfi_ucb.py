"""Tests for the friend's RandomForest model integration (rfi_ucb)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import SimConfig
from app.main import app
from app.sim.engine import run_simulation
from app.sim.schedulers.rfi_ucb import RFIUCBScheduler
from app.train import ARTIFACTS_DIR

client = TestClient(app)
ARTIFACT = ARTIFACTS_DIR / "band_activity_model.pkl"


def _scheduler():
    if not ARTIFACT.exists():
        pytest.skip("band_activity_model.pkl not integrated (requires scikit-learn 1.9)")
    return RFIUCBScheduler(10, artifact=str(ARTIFACT), seed=1)


def test_feature_matrix_shape():
    s = _scheduler()
    for t in range(12):
        s.tick(t)
        band = s.select(t)
        s.update(band, bool((t + band) % 3 == 0), t)
    F = s._features()
    assert F.shape == (10, 4)
    assert 0.0 <= F.min() <= F.max() <= 1.001


def test_model_loaded_and_predicts():
    s = _scheduler()
    assert s.model is not None
    import joblib

    m = joblib.load(str(ARTIFACT))
    p = m.predict_proba(np.zeros((1, 4)))[:, 1][0]
    assert 0.0 <= p <= 1.0


def test_full_run_with_friend_model():
    r = run_simulation(SimConfig(n_bands=10, n_steps=150, seed=42, scenario="switch", scheduler="rfi_ucb"))
    sm = r["smart"]["metrics"]
    assert sm["interception_ratio"] >= 0
    assert sm["avg_reward"] > -1
    assert "priorities" in r["smart"]


def test_rfi_ucb_cuts_wasted_scans():
    """On stable, the ML prior should hit as often as possible (high reward)."""
    r = run_simulation(SimConfig(n_bands=10, n_steps=200, seed=2024, scenario="stable", scheduler="rfi_ucb"))
    assert r["smart"]["metrics"]["hits"] >= 40  # dwells on the hot band


def test_api_registered_and_runs():
    scheds = {s["id"] for s in client.get("/api/schedulers").json()["schedulers"]}
    assert "rfi_ucb" in scheds
    models = {m["scheduler"] for m in client.get("/api/models").json()["models"]}
    assert "rfi_ucb" in models
    r = client.post("/api/simulate", json={"scenario": "switch", "scheduler": "rfi_ucb", "n_steps": 60, "seed": 9})
    assert r.status_code == 200
    assert r.json()["smart"]["metrics"]["hits"] >= 0