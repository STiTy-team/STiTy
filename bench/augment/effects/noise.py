import numpy as np

from ..assets import load
from .base import PlaceEffect, match_loudness


class Noise(PlaceEffect):
    NAME = "noise"

    def apply(self, samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        path = self.pick_file(rng)
        noise = load(path)
        start = int(rng.integers(len(noise)))
        noise = np.resize(np.roll(noise, -start), len(samples))
        level = self.cfg.level.sample(rng)
        mixed = samples + match_loudness(noise, samples, level)
        return mixed, {
            "file": self.assets.name(path),
            "start_sample": start,
            "level": round(level, 4),
        }
