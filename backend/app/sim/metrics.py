from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunMetrics:
    interception_ratio: float = 0.0      # % of transmission bursts intercepted
    avg_intercept_time: float = 0.0      # avg delay (steps) start -> first hit
    miss_count: int = 0                  # bursts never detected
    probability_of_detection: float = 0.0
    probability_of_false_alarm: float = 0.0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    hits: int = 0
    scans: int = 0
    adaptation_speed: float | None = None  # steps to re-prioritize after change
    correct_predictions_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "interception_ratio": round(self.interception_ratio * 100, 2),
            "avg_intercept_time": round(self.avg_intercept_time, 2),
            "miss_count": self.miss_count,
            "probability_of_detection": round(self.probability_of_detection, 4),
            "probability_of_false_alarm": round(self.probability_of_false_alarm, 4),
            "total_reward": round(self.total_reward, 2),
            "avg_reward": round(self.avg_reward, 4),
            "hits": self.hits,
            "scans": self.scans,
            "adaptation_speed": self.adaptation_speed,
            "correct_predictions_pct": round(self.correct_predictions_pct * 100, 2),
        }


def compute_metrics(
    truth_bool: np.ndarray,
    scan_log: list[dict],
    events: list[dict],
    reward_per_hit: float = 1.0,
    reward_per_miss: float = -0.5,
    noise_pfa: float = 0.0,
) -> RunMetrics:
    """Score one scanner against the ground truth + its own scan log.

    truth_bool: (T, B) boolean grid. scan_log: per-step entry dict with
    {'t', 'band', 'hit', 'snr'}. events: scenario transitions (for adaptation).
    """
    m = RunMetrics()
    T, B = truth_bool.shape
    scans = len(scan_log)
    m.scans = scans

    # ---- burst-segmented interception ---------------------------------------
    segments = _active_segments(truth_bool)
    detected_segs: set[tuple[int, int]] = set()
    first_hits: list[int] = []
    for e in scan_log:
        if e["hit"]:
            m.hits += 1
            m.total_reward += reward_per_hit
            seg = _segment_for(segments, int(e["band"]), int(e["t"]))
            if seg is not None and seg not in detected_segs:
                detected_segs.add(seg)
                first_hits.append(int(e["t"]) - seg[1])
        else:
            m.total_reward += reward_per_miss

    m.interception_ratio = len(detected_segs) / len(segments) if segments else 0.0
    m.miss_count = len(segments) - len(detected_segs)
    m.avg_intercept_time = float(np.mean(first_hits)) if first_hits else 0.0
    m.avg_reward = m.total_reward / max(1, m.scans)

    # ---- Pd / Pfa -----------------------------------------------------------
    scanned_on_active = sum(
        1 for e in scan_log if bool(truth_bool[e["t"], e["band"]])
    )
    m.probability_of_detection = (
        sum(1 for e in scan_log if e["hit"] and truth_bool[e["t"], e["band"]])
        / max(1, scanned_on_active)
    )
    scanned_empty = sums = sum(1 for e in scan_log if not truth_bool[e["t"], e["band"]])
    if scanned_empty:
        false_hits = sum(
            1 for e in scan_log if e["hit"] and not truth_bool[e["t"], e["band"]]
        )
        m.probability_of_false_alarm = false_hits / scanned_empty

    # ---- correct prediction % (would-be scan aligned with truth) --------------
    if scans:
        aligned = sum(1 for e in scan_log if bool(truth_bool[e["t"], e["band"]]))
        m.correct_predictions_pct = aligned / scans

    # ---- adaptation speed -----------------------------------------------------
    for ev in events:
        if ev["type"] != "change" and ev["bands_on"]:
            break
    for ev in events:
        if ev["bands_on"] and ev["type"] in ("change", "surprise"):
            t0 = int(ev["t"])
            new_bands = {int(b) for b in ev["bands_on"]}
            for e in scan_log:
                if e["t"] >= t0 and int(e["band"]) in new_bands and e["hit"]:
                    m.adaptation_speed = float(e["t"] - t0)
                    return m
            break
    return m


def _active_segments(truth_bool: np.ndarray) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []  # (band, start, end)
    T, B = truth_bool.shape
    for b in range(B):
        col = truth_bool[:, b]
        start = None
        for t in range(T):
            on = bool(col[t])
            if on and start is None:
                start = t
            if (not on or t == T - 1) and start is not None:
                end = t if not on else t
                segs.append((b, start, end))
                start = None
    return segs


def _segment_for(segs: list[tuple[int, int, int]], band: int, t: int):
    for (b, s, e) in segs:
        if b == band and s <= t <= e:
            return (b, s, e)
    return None