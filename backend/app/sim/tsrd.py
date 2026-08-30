"""TSRD-derived realistic emission profiles for the 10 reference bands.

The Turing Synthetic Radar Dataset (``data_stats.csv``) holds per-scan
aggregated PDW statistics (n_pulses, emitters, frequency, pulse width, AoA,
amplitude). This module maps each of the 10 frequency bands to a real TSRD
scan configuration and shapes the synthesised emitter behaviour from those
statistics, so the "live data emission" shown on the dashboard is grounded
in a real radar dataset rather than pure noise.

A bundled copy of the dataset lives at ``backend/data/tsrd_data_stats.csv``.
If it is missing the module falls back to the embedded ``DEFAULT_PROFILES``
table (extracted from the same file), so the app still runs offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .emitter import Emitter

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_BUNDLED_CSV = _DATA_DIR / "tsrd_data_stats.csv"
_USER_CSV = Path(r"C:\Users\vedhp\OneDrive\Desktop\Smart_scan\data\tsrd_data\scan\val_scan\data_stats.csv")

# Enabled features used from the per-scan statistics.
_COLS = [
    "n_pulses",
    "n_emitters",
    "n_types",
    "mean_Frequency",
    "max_Frequency",
    "min_Frequency",
    "mean_PulseWidth",
    "max_PulseWidth",
    "mean_AoA",
    "mean_Amplitude",
    "max_Amplitude",
]

# Mean-over-occurrences of config_0..config_9 extracted from data_stats.csv —
# offline fallback so the demo never depends on the external folder.
DEFAULT_PROFILES: dict[str, dict] = {
    "config_0": {"n_pulses": 53255.5, "n_emitters": 14.5, "n_types": 15.0, "mean_Frequency": 4861.93, "max_Frequency": 11995.92, "min_Frequency": 149.77, "mean_PulseWidth": 18.55, "max_PulseWidth": 106.43, "mean_AoA": 62.83, "mean_Amplitude": -85.3, "max_Amplitude": -39.74},
    "config_1": {"n_pulses": 70318.5, "n_emitters": 34.0, "n_types": 34.0, "mean_Frequency": 6145.66, "max_Frequency": 13734.62, "min_Frequency": 9.96, "mean_PulseWidth": 16.95, "max_PulseWidth": 350.13, "mean_AoA": 45.81, "mean_Amplitude": -89.87, "max_Amplitude": -6.41},
    "config_2": {"n_pulses": 192327.0, "n_emitters": 75.5, "n_types": 51.0, "mean_Frequency": 4328.36, "max_Frequency": 11405.46, "min_Frequency": 5.91, "mean_PulseWidth": 17.38, "max_PulseWidth": 351.65, "mean_AoA": 57.77, "mean_Amplitude": -85.01, "max_Amplitude": 2.61},
    "config_3": {"n_pulses": 181828.0, "n_emitters": 63.0, "n_types": 51.0, "mean_Frequency": 5213.65, "max_Frequency": 16066.22, "min_Frequency": 21.85, "mean_PulseWidth": 13.29, "max_PulseWidth": 353.05, "mean_AoA": 40.31, "mean_Amplitude": -82.51, "max_Amplitude": -4.31},
    "config_4": {"n_pulses": 39180.0, "n_emitters": 37.0, "n_types": 37.0, "mean_Frequency": 3468.12, "max_Frequency": 10508.22, "min_Frequency": 14.98, "mean_PulseWidth": 6.88, "max_PulseWidth": 345.19, "mean_AoA": -32.45, "mean_Amplitude": -93.45, "max_Amplitude": -10.95},
    "config_5": {"n_pulses": 28863.5, "n_emitters": 4.0, "n_types": 6.0, "mean_Frequency": 5058.97, "max_Frequency": 9402.62, "min_Frequency": 388.12, "mean_PulseWidth": 8.85, "max_PulseWidth": 107.3, "mean_AoA": -59.24, "mean_Amplitude": -80.25, "max_Amplitude": -31.79},
    "config_6": {"n_pulses": 144722.5, "n_emitters": 62.5, "n_types": 45.0, "mean_Frequency": 4855.74, "max_Frequency": 10012.6, "min_Frequency": 337.36, "mean_PulseWidth": 2.07, "max_PulseWidth": 213.44, "mean_AoA": 33.86, "mean_Amplitude": -91.35, "max_Amplitude": -21.14},
    "config_7": {"n_pulses": 89572.5, "n_emitters": 39.0, "n_types": 33.0, "mean_Frequency": 4967.6, "max_Frequency": 11998.06, "min_Frequency": 149.95, "mean_PulseWidth": 11.06, "max_PulseWidth": 53.07, "mean_AoA": 51.8, "mean_Amplitude": -83.89, "max_Amplitude": -19.39},
    "config_8": {"n_pulses": 32765.0, "n_emitters": 16.0, "n_types": 19.0, "mean_Frequency": 3714.61, "max_Frequency": 11964.19, "min_Frequency": 9.43, "mean_PulseWidth": 42.83, "max_PulseWidth": 349.2, "mean_AoA": 10.97, "mean_Amplitude": -76.29, "max_Amplitude": 13.99},
    "config_9": {"n_pulses": 32600.5, "n_emitters": 15.5, "n_types": 19.0, "mean_Frequency": 4119.64, "max_Frequency": 11986.05, "min_Frequency": 425.98, "mean_PulseWidth": 1.99, "max_PulseWidth": 211.32, "mean_AoA": 18.76, "mean_Amplitude": -89.71, "max_Amplitude": -34.68},
}

_PROFILE_IDS = ["config_%d" % i for i in range(10)]


def _csv_candidates() -> list[Path]:
    return [_BUNDLED_CSV, _USER_CSV]


def load_profiles() -> dict[str, dict]:
    """Best-effort load of the per-band profiles from the TSRD CSV."""
    try:
        import pandas as pd
    except ImportError:
        return dict(DEFAULT_PROFILES)
    for path in _csv_candidates():
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            sel = df[df["id"].isin(_PROFILE_IDS)]
            if len(sel) == 0:
                continue
            g = sel.groupby("id")[_COLS].mean()
            return {key: {c: float(g.loc[key, c]) for c in _COLS} for key in g.index}
        except Exception:
            continue
    return dict(DEFAULT_PROFILES)


def band_profiles(n_bands: int = 10) -> list[dict]:
    """Profiles for bands 0..n_bands-1 (one TSRD config per band)."""
    profs = load_profiles()
    out: list[dict] = []
    for b in range(n_bands):
        key = _PROFILE_IDS[b % len(_PROFILE_IDS)]
        if key not in profs:
            continue
        out.append({"band": b, "id": key, **profs[key]})
    return out


def _duty_for(profile: dict, max_pulses: float) -> float:
    return float(np.clip(0.08 + 0.85 * profile["n_pulses"] / max(max_pulses, 1.0), 0.08, 0.92))


def tsrd_emitters(n_bands: int = 10, n_steps: int = 300, seed: int = 42) -> list[Emitter]:
    """Synthesise per-band emitters whose duty/periodicity/lifetimes mirror
    the TSRD statistics for that band."""
    profs = band_profiles(n_bands)
    if not profs:
        return [Emitter(name="e_fallback", bands=[3], interval=1, rng_seed=seed)]
    max_pulses = max(p["n_pulses"] for p in profs) or 1.0
    emitters: list[Emitter] = []
    for p in profs:
        b = p["band"]
        duty = _duty_for(p, max_pulses)
        count = int(np.clip(round(max(1.0, p["n_emitters"] / 18.0)), 1, 4))
        regime = b % 3  # 0 always-on · 1 first half · 2 second half (switch-in)
        for e in range(count):
            name = f"tsrd_{p['id']}_b{b}_e{e}"
            start, end = 0, None
            if regime == 1:
                end = int(n_steps * 0.55)
            elif regime == 2:
                start = int(n_steps * 0.4)
            rng_seed = seed + b * 97 + e * 13
            hot = duty > 0.55
            if hot and e == 0:
                em = Emitter(name, bands=[b], interval=1, start=start, end=end,
                             burst=min(0.95, duty + 0.15), rng_seed=rng_seed)
            elif hot:
                em = Emitter(name, bands=[b], interval=3, phase=e % 3, start=start, end=end,
                             burst=min(0.9, duty), rng_seed=rng_seed)
            else:
                em = Emitter(name, bands=[b], interval=4 + e, phase=e, start=start, end=end,
                             burst=min(0.85, duty + 0.2), rng_seed=rng_seed)
            emitters.append(em)
    # Late-appearing surprise emitter on the quietest band (realistic pop-up).
    quiet = min(profs, key=lambda p: p["n_pulses"])
    qb = quiet["band"]
    emitters.append(
        Emitter(name="tsrd_surprise", bands=[qb], interval=1, start=int(n_steps * 0.62), end=n_steps,
                burst=0.55, rng_seed=seed + 9091)
    )
    return emitters


def tsrd_band_meta(n_bands: int = 10) -> list[dict]:
    """Human-readable per-band reference metadata for the dashboard."""
    profs = band_profiles(n_bands)
    if not profs:
        return [{"band": b} for b in range(n_bands)]
    max_pulses = max(p["n_pulses"] for p in profs) or 1.0
    out = []
    for p in profs:
        out.append(
            {
                "band": p["band"],
                "config_id": p["id"],
                "center_mhz": round(p["mean_Frequency"], 1),
                "freq_min_mhz": round(p["min_Frequency"], 1),
                "freq_max_mhz": round(p["max_Frequency"], 1),
                "n_pulses": int(p["n_pulses"]),
                "n_emitters": int(round(p["n_emitters"])),
                "n_types": int(round(p["n_types"])),
                "pulse_width_us": round(p["mean_PulseWidth"], 2),
                "aoa_deg": round(p["mean_AoA"], 1),
                "amplitude_dbm": round(p["mean_Amplitude"], 1),
                "duty": round(_duty_for(p, max_pulses), 2),
            }
        )
    return out