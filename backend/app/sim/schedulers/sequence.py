from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baseline import BaseScheduler

NORM = {
    "n_bands": 0,
}  # populated from artifact meta at load time


class SequenceScheduler(BaseScheduler):
    """Sequence-aware LSTM: predicts the periodic/hopping emitter's next band.

    The model was trained offline on emitter chronologies. At runtime it biases
    the scan choice towards its confident predictions while an EWMA fallback
    keeps covering the rest of the spectrum.
    """

    name = "sequence"

    def __init__(self, n_bands: int, artifact_dir: str | Path, seed: int = 0, alpha: float = 0.3):
        super().__init__(n_bands, seed)
        self.alpha = float(alpha)
        self.q = np.zeros(self.n, dtype=float)
        self.counts = np.zeros(self.n, dtype=float)
        self.history: list[np.ndarray] = []  # list of (one-hot band, active)
        self._dir = Path(artifact_dir)
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("torch required for sequence scheduler") from e
        meta, self._model = self._load()
        self.window = int(meta["window"])
        self.threshold = float(meta.get("threshold", 0.55))

    def _load(self) -> tuple[dict, object]:
        import torch

        from app.train.seq_model import SeqModel

        meta_path = Path(self._dir) / "meta.json"
        model_path = Path(self._dir) / "sequence.pt"
        if not (meta_path.exists() and model_path.exists()):
            raise RuntimeError(f"sequence artifact missing in {self._dir}")
        meta = json.loads(meta_path.read_text())
        model = SeqModel(
            n_bands=int(meta.get("n_bands", self.n)),
            window=int(meta.get("window", 20)),
            hidden=int(meta.get("hidden", 64)),
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        return meta, model

    # ------------------------------------------------------------- scheduler
    def tick(self, t: int) -> None:
        super().tick(t)
        self.q = self.q * (1.0 - self.alpha)

    def update(self, band: int, hit: bool, t: int) -> None:
        self.counts[band] += 1.0
        if hit:
            self.q[band] += self.alpha
        onehot = np.zeros(self.n, dtype=float)
        onehot[band] = 1.0
        self.history.append(np.concatenate([onehot, np.array([1.0 if hit else 0.0])]))

    def _features(self) -> np.ndarray:
        import torch

        tail = self.history[-self.window :]
        rows: list[np.ndarray] = []
        for _ in range(self.window - len(tail)):  # left-pad with zero slots
            rows.append(np.zeros(self.n + 2))  # one-hot(n) + active + none-flag
        for row in tail:  # stored as (one-hot, active); none-flag = 1 when quiet
            none_flag = np.array([0.0]) if row[-1] > 0.5 else np.array([1.0])
            rows.append(np.concatenate([row, none_flag]))
        arr = np.stack(rows[-self.window :])
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1, W, n+1)

    def select(self, t: int) -> int:
        import torch

        if len(self.history) >= self.window and self.rng.random() < 0.9:
            with torch.no_grad():
                logits = self._model(self._features())
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            best = int(np.argmax(probs))
            conf = float(probs[best])
            if best < self.n and conf >= self.threshold and self.rng.random() < conf:
                return best  # confident prediction on a real band
            # 'quiet' class or low confidence -> fall back to activity estimate
        return int(np.argmax(self.q)) if np.any(self.q > 0) else int(self.rng.integers(self.n))