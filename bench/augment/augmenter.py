import hashlib
from pathlib import Path

import numpy as np

from .config import AugmentConfig
from .effects import ORDER, Effect


class Augmenter:
    def __init__(self, effects: list[Effect]):
        self.effects = effects

    def apply(self, samples: np.ndarray, *, key: str) -> tuple[np.ndarray, dict]:
        choices = {}
        for effect in self.effects:
            rng = _rng(key, effect.NAME, effect.fingerprint)
            samples, choices[effect.NAME] = effect.apply(samples, rng)
        peak = float(np.max(np.abs(samples), initial=0.0))
        if peak > 1.0:
            samples = samples / peak
        return samples.astype(np.float32), choices


def item_key(dataset: str, item_id: str) -> str:
    return f"{dataset}/{item_id}"


def build(cfg: AugmentConfig | None, *, data_root: Path) -> Augmenter:
    if cfg is None:
        return Augmenter([])
    return Augmenter(
        [
            kind.create(spec, data_root=data_root)
            for kind in ORDER
            if (spec := getattr(cfg, kind.NAME)) is not None
        ]
    )


def _rng(*parts: str) -> np.random.Generator:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
