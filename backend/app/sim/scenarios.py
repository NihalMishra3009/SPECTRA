from __future__ import annotations

from .emitter import Emitter

# --------------------------------------------------------------------------
# The 12 operating scenarios from the problem document.
# Each returns the emitter set that defines the ground-truth RF environment.
# --------------------------------------------------------------------------


def s1_stable(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Stable pattern: fixed hot bands stay active the whole run."""
    return [Emitter(name="e_stable_a", bands=[1, 3, 5, 7], interval=1)]


def s2_switch(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Pattern switch: hot bands change mid-run (no prior warning)."""
    mid = steps // 2
    return [
        Emitter(name="e_first", bands=[1, 3, 5], start=0, end=mid - 1),
        Emitter(name="e_second", bands=[2, 4], start=mid),
    ]


def s3_single_dominant(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Single dominant band with a little background elsewhere."""
    return [Emitter(name="e_dominant", bands=[5], interval=1)]


def s4_multi_equal(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Many equally active bands."""
    return [Emitter(name="e_multi", bands=list(range(0, n)), interval=1)]


def s5_random(n: int = 10, steps: int = 300) -> list[Emitter]:
    """No structure — random sporadic activity everywhere."""
    return [
        Emitter(name=f"e_rand_{i}", bands=[i], interval=1, burst=0.08, rng_seed=100 + i)
        for i in range(n)
    ]


def s6_surprise(n: int = 10, steps: int = 300) -> list[Emitter]:
    """New emitter appears mid-run after being silent the whole time."""
    appears = int(steps * 0.55)
    return [
        Emitter(name="e_known", bands=[2], interval=1),
        Emitter(name="e_surprise", bands=[7], start=appears, end=steps),
    ]


def s7_periodic(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Periodic emitter: fixed scan interval to be timed & predicted."""
    return [
        Emitter(name="e_base", bands=[3], interval=1),
        Emitter(name="e_periodic", bands=[8], interval=12, phase=3),
    ]


def s13_periodic_only(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Pure periodic emitters — timing prediction demo (no constant carrier)."""
    return [
        Emitter(name="e_scan_a", bands=[8], interval=12, phase=3),
        Emitter(name="e_scan_b", bands=[2], interval=7, phase=5),
    ]


def s8_hopper(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Frequency-hopping agile emitter cycling a pseudorandom band set."""
    return [
        Emitter(name="e_base", bands=[1, 4], interval=1),
        Emitter(name="e_hopper", bands=[0], interval=2, hop_bands=[6, 5, 9, 2]),
    ]


def s9_burst(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Intermittent/burst emitter — silence is not 'dead'."""
    return [
        Emitter(name="e_base", bands=[2, 5], interval=1),
        Emitter(name="e_bursty", bands=[8], interval=1, burst=0.35, rng_seed=777),
    ]


def s10_multi_sim(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Multiple simultaneous emitters under a single-channel receiver."""
    return [
        Emitter(name="e_a", bands=[1], interval=1),
        Emitter(name="e_b", bands=[3], interval=1),
        Emitter(name="e_c", bands=[5], interval=1),
        Emitter(name="e_d", bands=[7], interval=1),
    ]


def s11_noise(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Noise / false alarms — avoid learning from false positives."""
    return [Emitter(name="e_base", bands=[1, 3], interval=1)]


def s12_threat(n: int = 10, steps: int = 300) -> list[Emitter]:
    """Threat-priority weighting where threat intelligence exists."""
    return [
        Emitter(name="e_low", bands=[1, 6], interval=1, threat_weight=1.0),
        Emitter(name="e_high", bands=[4], interval=3, phase=0, threat_weight=5.0),
    ]


SCENARIOS: dict[str, dict] = {
    "stable": {
        "label": "Stable pattern",
        "desc": "Fixed hot bands stay active (Scenario 1)",
        "fun": s1_stable,
    },
    "switch": {
        "label": "Pattern switch",
        "desc": "Hot bands change mid-run (Scenario 2)",
        "fun": s2_switch,
    },
    "single_dominant": {
        "label": "Single dominant band",
        "desc": "One hot band dominates (Scenario 3)",
        "fun": s3_single_dominant,
    },
    "multi_equal": {
        "label": "Multiple equal bands",
        "desc": "Even priority across all bands (Scenario 4)",
        "fun": s4_multi_equal,
    },
    "random": {
        "label": "Random / no structure",
        "desc": "Uniform coverage, no false learning (Scenario 5)",
        "fun": s5_random,
    },
    "surprise": {
        "label": "Surprise emitter",
        "desc": "New emitter appears mid-run (Scenario 6)",
        "fun": s6_surprise,
    },
    "periodic": {
        "label": "Periodic emitter",
        "desc": "Fixed interval to learn & time (Scenario 7)",
        "fun": s7_periodic,
    },
    "periodic_only": {
        "label": "Periodic timing demo",
        "desc": "Pure periodic emitters for LSTM timing prediction",
        "fun": s13_periodic_only,
    },
    "hopper": {
        "label": "Frequency hopper",
        "desc": "Agile hop sequence (Scenario 8)",
        "fun": s8_hopper,
    },
    "burst": {
        "label": "Bursty emitter",
        "desc": "Intermittent, don't call it dead (Scenario 9)",
        "fun": s9_burst,
    },
    "multi_sim": {
        "label": "Multiple emitters",
        "desc": "Simultaneous, one receiver channel (Scenario 10)",
        "fun": s10_multi_sim,
    },
    "noise": {
        "label": "Noise / false alarms",
        "desc": "Robust to noise spikes (Scenario 11)",
        "fun": s11_noise,
    },
    "threat": {
        "label": "Threat priority",
        "desc": "Priority weighting over raw activity (Scenario 12)",
        "fun": s12_threat,
    },
    "stable_switch_surprise": {
        "label": "Demo: Stable + Switch + Surprise",
        "desc": "The 3-stage crowd-pleaser (Stages 1-3)",
        "fun": s2_switch,  # reuse switch: it already contains the surprise & change
    },
}


def get_emitters(scenario_id: str, n_bands: int, n_steps: int) -> list[Emitter]:
    key = SCENARIOS.get(scenario_id, SCENARIOS["stable_switch_surprise"])
    return key["fun"](n_bands, n_steps)


def scenario_catalog() -> list[dict]:
    return [
        {"id": k, "label": v["label"], "desc": v["desc"]} for k, v in SCENARIOS.items()
    ]