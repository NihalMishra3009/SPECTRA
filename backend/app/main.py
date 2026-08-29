from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router

app = FastAPI(
    title="SPECTRA — Smart Scan for Electronic Warfare",
    version="1.0.0",
    description="ML-based ES receiver scheduler + simulator (SIH Problem 26055)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {"name": "SPECTRA EW Scheduler", "docs": "/docs", "ws": "/ws/simulate"}