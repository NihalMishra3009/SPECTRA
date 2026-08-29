import os
from pathlib import Path

# Model artifacts live in the repo-level `model/artifacts/` folder so all
# trained models + their metadata are organized in ONE place.
MODEL_ROOT = Path(os.environ.get("SPECTRA_MODEL_DIR", Path(__file__).resolve().parents[3] / "model"))
ARTIFACTS_DIR = Path(os.environ.get("SPECTRA_ARTIFACTS_DIR", MODEL_ROOT / "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "rl_dqn": {"file": "dqn.zip", "meta": "dqn_meta.json", "curves": "dqn_curves.json"},
    "rl_ppo": {"file": "ppo.zip", "meta": "ppo_meta.json", "curves": "ppo_curves.json"},
    "bandit_baseline": {"file": "bandit_baseline.json"},
    "sequence": {"file": "sequence.pt", "meta": "meta.json", "curves": "sequence_curves.json"},
    "dataset_rfi": {"file": "turing_model.pkl", "meta": "turing_model_meta.json"},
    "rfi_ucb": {"file": "band_activity_model.pkl", "meta": "rfi_ucb_meta.json"},
}