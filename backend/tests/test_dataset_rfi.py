"""Tests for the dataset-trained RFI scheduler integration."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import SimConfig
from app.main import app
from app.sim.engine import make_scheduler, run_simulation
from app.sim.schedulers.dataset_rfi import (
    DatasetRFIScheduler,
    build_features,
    load_model,
    predict_scores,
)
from app.train import ARTIFACTS_DIR

client = TestClient(app)


def test_build_features_shape_and_bounds():
    hits = np.array([0.5, 0.1, 0.0])
    misses = np.array([0.0, 12.0, 3.0])
    f = build_features(hits, misses, t=60, n_steps=300)
    assert f.shape == (2 * 3 + 1,)
    assert f[0] == 0.5
    assert 0.9 < f[4] < 1.0  # tanh(12/5) — misses[1]
    assert f[-1] == pytest.approx(0.2)


def test_load_model_missing_returns_none(tmp_path):
    assert load_model(tmp_path / "nope.pkl") is None


def test_predict_scores_handles_proba_and_plain():
    class A:  # predict_proba
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])

    class B:  # predict only
        def predict(self, X):
            return np.array([[0.7, 0.3]])

    assert predict_scores(A(), np.zeros((1, 3))).shape == (1, 2)
    assert predict_scores(B(), np.zeros((1, 3)))[0, 0] == 0.7


def test_dataset_rfi_scheduler_serves_demo_artifact():
    artifact = ARTIFACTS_DIR / "turing_model.pkl"
    if not artifact.exists():
        pytest.skip("demo artifact not built (run integrate_dataset_model --build-demo)")
    s = DatasetRFIScheduler(10, artifact=str(artifact), n_steps=300)
    for t in range(50):
        s.tick(t)
        band = s.select(t)
        assert 0 <= band < 10
        s.update(band, hit=bool((t + band) % 3 == 0), t=t)


def test_dataset_rfi_without_artifact_falls_back():
    s = DatasetRFIScheduler(10, artifact=None, n_steps=300)
    assert s.model is None
    for t in range(20):
        s.tick(t)
        assert 0 <= s.select(t) < 10


def test_make_scheduler_registered():
    c = SimConfig(scenario="switch", scheduler="dataset_rfi")
    sched = make_scheduler("dataset_rfi", c, seed=1, n_bands=10)
    assert isinstance(sched, DatasetRFIScheduler)
    assert sched.model is not None  # demo artifact present


def test_full_run_with_dataset_rfi():
    r = run_simulation(SimConfig(n_bands=10, n_steps=120, seed=7, scenario="switch", scheduler="dataset_rfi"))
    assert r["smart"]["metrics"]["interception_ratio"] >= 0
    assert "priorities" in r["smart"]  # belief heatmap still tracked


def test_api_exposes_dataset_rfi_and_runs():
    scheds = {s["id"] for s in client.get("/api/schedulers").json()["schedulers"]}
    assert "dataset_rfi" in scheds

    models = {m["scheduler"] for m in client.get("/api/models").json()["models"]}
    assert "dataset_rfi" in models

    r = client.post(
        "/api/simulate",
        json={"scenario": "switch", "scheduler": "dataset_rfi", "n_steps": 60, "seed": 5},
    )
    assert r.status_code == 200
    assert r.json()["smart"]["metrics"]["hits"] >= 0