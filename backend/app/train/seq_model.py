from __future__ import annotations

import torch
import torch.nn as nn


class SeqModel(nn.Module):
    """LSTM that predicts the *next band* an emitter will be on (+ 'quiet')."""

    def __init__(self, n_bands: int, window: int, hidden: int = 64):
        super().__init__()
        self.n_bands = int(n_bands)
        self.window = int(window)
        self.hidden = int(hidden)
        self.in_dim = n_bands + 2  # one-hot band + active + quiet flags
        self.lstm = nn.LSTM(self.in_dim, self.hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(self.hidden, n_bands + 1)  # +1 = 'quiet'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])