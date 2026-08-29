"""Loader for the 'Turing Synthetic Radar Dataset' (HuggingFace).

Used for external validation (Phase 5 of the roadmap). The primary training
signal remains the custom RF simulator; this module is a clean seam to swap
ground-truth sources later.

Dataset link from the SIH problem statement:
    https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset

Optional dependency: `pip install datasets pandas`
"""
from __future__ import annotations

DEFAULT_ID = "alan-turing-institute/turing-synthetic-radar-dataset"


def available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def load(dataset_id: str = DEFAULT_ID, split: str = "train", max_rows: int | None = None):
    if not available():
        raise RuntimeError("pip install datasets pandas to use the Turing loader")
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds


def schema(dataset_id: str = DEFAULT_ID, rows: int = 5) -> dict:
    ds = load(dataset_id=dataset_id, max_rows=rows)
    return {
        "num_rows_total": sum(1 for _ in ds),
        "columns": list(ds.column_names),
        "features": {k: str(v) for k, v in ds.features.items()},
        "sample": ds[: min(rows, len(ds))],
    }


def build_truth_grid(dataset_id: str = DEFAULT_ID, n_bands: int = 10):
    """Phase-5 placeholder: map Turing emitter pulses onto our band/time grid."""
    raise NotImplementedError(
        "Phase 5: map Turing dataset pulses -> (band, time) grid and re-run validation."
    )