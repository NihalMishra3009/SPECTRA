from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "rl_dqn": {"file": "dqn.zip", "meta": "dqn_meta.json", "curves": "dqn_curves.json"},
    "rl_ppo": {"file": "ppo.zip", "meta": "ppo_meta.json", "curves": "ppo_curves.json"},
    "bandit_baseline": {"file": "bandit_baseline.json"},
    "sequence": {"file": "sequence.pt", "meta": "meta.json", "curves": "sequence_curves.json"},
}