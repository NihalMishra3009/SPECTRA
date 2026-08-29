from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from ..config import DEFAULT_DEMO, SCHEDULERS, SimConfig
from ..db import get_run, list_runs, save_run, stats
from ..sim.engine import run_simulation, run_writer
from ..sim.scenarios import scenario_catalog
from ..train import ARTIFACTS_DIR, MODEL_REGISTRY

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "SPECTRA EW scheduler"}


@router.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": scenario_catalog()}


@router.get("/api/schedulers")
def schedulers() -> dict:
    return {"schedulers": [{"id": k, "desc": v} for k, v in SCHEDULERS.items()],
            "defaults": DEFAULT_DEMO}


@router.get("/api/models")
def models() -> dict:
    items = []
    for key, spec in MODEL_REGISTRY.items():
        meta_file = spec.get("meta")
        if meta_file and (ARTIFACTS_DIR / meta_file).exists():
            meta = json.loads((ARTIFACTS_DIR / meta_file).read_text())
            items.append(
                {
                    "scheduler": key,
                    "present": True,
                    "meta": meta,
                    "curves_file": spec.get("curves"),
                }
            )
        elif "file" in spec and (ARTIFACTS_DIR / spec["file"]).exists():
            items.append({"scheduler": key, "present": True, "meta": None})
        else:
            items.append({"scheduler": key, "present": False, "meta": None})
    return {"models": items}


@router.get("/api/curves/{name}")
def curves(name: str) -> JSONResponse:
    f = ARTIFACTS_DIR / name
    if f.exists():
        return JSONResponse(json.loads(f.read_text()))
    return JSONResponse({"error": "curve not found"}, status_code=404)


@router.post("/api/simulate")
def simulate(cfg: SimConfig) -> dict:
    known = {s["id"] for s in scenario_catalog()}
    if cfg.scenario not in known:
        raise HTTPException(status_code=400, detail=f"unknown scenario: {cfg.scenario}")
    if cfg.scheduler not in SCHEDULERS:
        raise HTTPException(status_code=400, detail=f"unknown scheduler: {cfg.scheduler}")
    result = run_simulation(cfg)
    try:
        result["run_id"] = save_run(result)
    except Exception:
        result["run_id"] = None  # persistence is best-effort, never blocks a run
    return result


# --------------------------------------------------------------------------- #
#  database (recorded RF measurements)
# --------------------------------------------------------------------------- #
@router.get("/api/db/runs")
def db_runs(limit: int = 20) -> dict:
    return {"runs": list_runs(limit=max(1, min(limit, 200)))}


@router.get("/api/db/runs/{run_id}")
def db_run(run_id: str) -> dict:
    data = get_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return data


@router.get("/api/db/stats")
def db_stats() -> dict:
    return stats()


@router.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket) -> None:
    await ws.accept()
    try:
        cfg = SimConfig(**(await ws.receive_json()))
    except Exception:
        cfg = SimConfig(**DEFAULT_DEMO)
    try:
        for ev in run_writer(cfg):
            await ws.send_json(ev)
            await asyncio.sleep(0.02)
    except Exception:
        pass
    finally:
        await ws.close()