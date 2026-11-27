"""Language head over whisper encoder frames: attentive statistics pooling (attention-weighted
mean and standard deviation over the clip's own frames) and a small classifier. Trained by
core/research/lid_module/short_train.py on short crops of many languages."""

import torch
from torch import nn


class LidHead(nn.Module):
    def __init__(self, n_langs: int, dim: int = 768, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classify = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_langs)
        )
        self.config = {"n_langs": n_langs, "dim": dim, "hidden": hidden, "dropout": dropout}

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """frames [B, T, dim], mask [B, T] (True = the clip's frame) -> logits [B, n_langs]."""
        score = self.attn(frames).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        mean = (w * frames).sum(1)
        var = (w * (frames - mean.unsqueeze(1)) ** 2).sum(1)
        return self.classify(torch.cat([mean, var.clamp_min(1e-6).sqrt()], dim=-1))

    def save(self, path, langs: list[str], meta: dict) -> None:
        torch.save(
            {"state": self.state_dict(), "config": self.config, "langs": list(langs), "meta": meta},
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        blob = torch.load(path, map_location=map_location, weights_only=False)
        head = cls(**blob["config"])
        head.load_state_dict(blob["state"])
        return head, blob["langs"], blob.get("meta", {})
