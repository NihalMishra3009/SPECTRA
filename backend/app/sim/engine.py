from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import SimConfig
from .environment import RFEnvironment
from .metrics import RunMetrics, compute_metrics
from .receiver import Receiver
from .scenarios import get_emitters, scenario_catalog


def make_scheduler(scheduler_id: str, config: SimConfig, seed: int, n_bands: int):
    """Instantiates the requested scheduler (bandit / RL / sequence / baseline)."""
    from .schedulers import (  # local import avoids hard SB3/torch dependency at boot
        DatasetRFIScheduler,
        EpsilonGreedy,
        RlPolicyScheduler,
        RFIUCBScheduler,
        RoundRobin,
        SequenceScheduler,
        SlidingWindow,
        ThompsonSampling,
        UCB1,
    )

    if scheduler_id == "round_robin":
        return RoundRobin(n_bands, seed)
    if scheduler_id == "epsilon_greedy":
        return EpsilonGreedy(n_bands, alpha=config.alpha, epsilon=config.epsilon, floor=config.floor, seed=seed)
    if scheduler_id == "ucb1":
        return UCB1(n_bands, alpha=config.alpha, floor=config.floor, seed=seed)
    if scheduler_id == "thompson":
        return ThompsonSampling(n_bands, alpha=config.alpha, floor=config.floor, seed=seed)
    if scheduler_id in ("adaptive", "adaptive_window"):
        return SlidingWindow(n_bands, window=config.window, seed=seed, epsilon=config.epsilon)
    if scheduler_id in ("rl_dqn", "rl_ppo"):
        from ..train import ARTIFACTS_DIR

        file = {"rl_dqn": "dqn.zip", "rl_ppo": "ppo.zip"}[scheduler_id]
        artifact = str(ARTIFACTS_DIR / file)
        return RlPolicyScheduler(n_bands, artifact, seed=seed, alpha=config.alpha)
    if scheduler_id == "sequence":
        from ..train import ARTIFACTS_DIR

        return SequenceScheduler(n_bands, ARTIFACTS_DIR, seed=seed, alpha=config.alpha)
    if scheduler_id == "dataset_rfi":
        from ..train import ARTIFACTS_DIR

        return DatasetRFIScheduler(
            n_bands,
            artifact=str(ARTIFACTS_DIR / "turing_model.pkl"),
            seed=seed,
            alpha=config.alpha,
            epsilon=config.epsilon,
            floor=config.floor,
            n_steps=config.n_steps,
        )
    if scheduler_id == "rfi_ucb":
        from ..train import ARTIFACTS_DIR

        return RFIUCBScheduler(
            n_bands,
            artifact=str(ARTIFACTS_DIR / "band_activity_model.pkl"),
            seed=seed,
            alpha=config.alpha,
            floor=config.floor,
        )
    raise ValueError(f"unknown scheduler: {scheduler_id}")


# --------------------------------------------------------------------------- #
#  Single run helpers
# --------------------------------------------------------------------------- #
def _run_single(
    env: RFEnvironment,
    scheduler,
    reward_hit: float = 1.0,
    reward_miss: float = -0.5,
    noise_pfa: float = 0.0,
    priority_track: list | None = None,
) -> tuple[list[dict], RunMetrics]:
    rx = Receiver(env, snr_noise=noise_pfa)
    log: list[dict] = []
    for t in range(env.n_steps):
        scheduler.tick(t)
        band = int(scheduler.select(t))
        hit, snr = rx.observe(band, t)
        scheduler.update(band, hit, t)
        if priority_track is not None and hasattr(scheduler, "q"):
            priority_track.append(np.asarray(scheduler.q, dtype=float).copy())
        log.append(
            {
                "t": int(t),
                "band": band,
                "hit": bool(hit),
                "snr": round(float(snr), 3),
                "reward": reward_hit if hit else reward_miss,
            }
        )
    metrics = compute_metrics(env.ground_truth, log, env.events, reward_hit, reward_miss)
    return log, metrics


def _cumulative_ratio(log: list[dict], truth: np.ndarray, bands: int) -> list[float]:
    """Running interception % — segments started so far that have been hit."""
    starts: list[tuple[int, int]] = []
    seg_by_band: dict[int, list[tuple[int, int, int]]] = {}
    for b in range(bands):
        col = truth[:, b]
        s = None
        for t in range(len(col)):
            on = bool(col[t])
            if on and s is None:
                s = t
            if (not on or t == len(col) - 1) and s is not None:
                seg_by_band.setdefault(b, []).append((b, s, t if not on else t))
                starts.append((s, b))
                s = None
    starts.sort()
    detected: set[tuple[int, int]] = set()
    detect_t: dict[tuple[int, int], int] = {}
    for e in log:
        if e["hit"]:
            for spec in seg_by_band.get(e["band"], []):
                b, s, en = spec
                if s <= e["t"] <= en and (b, s) not in detect_t:
                    detect_t[(b, s)] = e["t"]

    idx = 0
    out: list[float] = []
    started = hit_n = 0
    for t in range(len(log)):
        while idx < len(starts) and starts[idx][0] <= t:
            started += 1
            idx += 1
        if started:
            hit_n = sum(1 for (b, s) in detect_t if s <= t)
        out.append(round(100.0 * hit_n / max(1, started), 2))
    return out


# --------------------------------------------------------------------------- #
#  Public entry point — used by REST, WS and tests
# --------------------------------------------------------------------------- #
def run_simulation(config: SimConfig) -> dict:
    emitters = get_emitters(config.scenario, config.n_bands, config.n_steps)
    env = RFEnvironment(
        n_bands=config.n_bands,
        n_steps=config.n_steps,
        emitters=emitters,
        noise=0.0,
        seed=config.seed,
        freq_start_ghz=config.freq_start_ghz,
        freq_end_ghz=config.freq_end_ghz,
    )

    # ---- fair comparison on the SAME environment ---------------------------
    base_log, base_metrics = _run_single(
        env, make_scheduler("round_robin", config, config.seed, config.n_bands)
    )
    smart = make_scheduler(config.scheduler, config, config.seed, config.n_bands)
    smart_q: list = []
    smart_log, smart_metrics = _run_single(env, smart, priority_track=smart_q)

    # cumulative curves for the dashboard charts
    base_curve = _cumulative_ratio(base_log, env.ground_truth, config.n_bands)
    smart_curve = _cumulative_ratio(smart_log, env.ground_truth, config.n_bands)
    for i, e in enumerate(base_log):
        e["ratio"] = base_curve[i]
    for i, e in enumerate(smart_log):
        e["ratio"] = smart_curve[i]

    # optional smart priority trajectory (belief heatmap)
    priorities = [row.tolist() for row in smart_q] if smart_q else None

    sc = scenario_catalog()
    sc_meta = next((s for s in sc if s["id"] == config.scenario), {"label": config.scenario})

    return {
        "config": config.model_dump(),
        "scenario_id": config.scenario,
        "scenario_label": sc_meta.get("label", config.scenario),
        "ground_truth": env.as_json_grid(),
        "band_edges_ghz": env.band_edges_ghz(),
        "activity_profile": env.activity_profile(),
        "events": env.events,
        "n_segments": len(
            [(s["start"], s["end"]) for s in env.active_segments()]
        ),
        "total_transmissions": env.total_transmissions(),
        "baseline": {
            "label": "Round-robin (Open-loop)",
            "scheduler": "round_robin",
            "log": base_log,
            "metrics": base_metrics.to_dict(),
        },
        "smart": {
            "label": smart.name,
            "scheduler": config.scheduler,
            "log": smart_log,
            "metrics": smart_metrics.to_dict(),
            "priorities": priorities,
        },
        "meta": {
            "n_bands": config.n_bands,
            "n_steps": config.n_steps,
            "seed": config.seed,
            "alpha": config.alpha,
            "epsilon": config.epsilon,
        },
    }


def run_writer(config: SimConfig):
    """Generator of per-step WS events for live telemetry."""
    emitters = get_emitters(config.scenario, config.n_bands, config.n_steps)
    env = RFEnvironment(
        n_bands=config.n_bands,
        n_steps=config.n_steps,
        emitters=emitters,
        noise=0.0,
        seed=config.seed,
        freq_start_ghz=config.freq_start_ghz,
        freq_end_ghz=config.freq_end_ghz,
    )
    from .schedulers import RoundRobin

    base = RoundRobin(config.n_bands, config.seed)
    smart = make_scheduler(config.scheduler, config, config.seed, config.n_bands)
    rxb = Receiver(env, snr_noise=0.0)
    rxs = Receiver(env, snr_noise=0.0)

    sw_tracker = _SwitchTracker(env)
    for t in range(config.n_steps):
        base.tick(t)
        smart.tick(t)
        bb = int(base.select(t))
        sb = int(smart.select(t))
        bh, bsnr = rxb.observe(bb, t)
        sh, ssnr = rxs.observe(sb, t)
        base.update(bb, bh, t)
        smart.update(sb, sh, t)
        yield {
            "t": t,
            "truth": env.ground_truth[t].tolist(),
            "baseline": {"band": bb, "hit": bh, "snr": round(float(bsnr), 3)},
            "smart": {"band": sb, "hit": sh, "snr": round(float(ssnr), 3)},
            "event": sw_tracker.step(t, env.ground_truth[t]),
        }


class _SwitchTracker:
    def __init__(self, env: RFEnvironment):
        self.prev = set(np.where(env.ground_truth[0])[0].tolist()) if env.n_steps else set()
        self.known: set[int] = set(self.prev)

    def step(self, t: int, truth_row: np.ndarray) -> dict | None:
        cur = set(np.where(truth_row)[0].tolist())
        surprise = sorted(cur - self.known)
        changed = cur != self.prev
        bands_on = sorted(cur - self.prev)
        self.known |= cur
        self.prev = cur
        if surprise or changed:
            return {
                "type": "surprise" if surprise else "change",
                "bands_on": sorted(surprise) if surprise else bands_on,
            }
        return None