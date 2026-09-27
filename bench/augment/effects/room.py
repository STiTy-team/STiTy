import numpy as np
from scipy.signal import fftconvolve

from ..assets import load
from .base import PlaceEffect, match_loudness


class Room(PlaceEffect):
    NAME = "room"

    def apply(self, samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        path = self.pick_file(rng)
        response = load(path)
        response = response[int(np.argmax(np.abs(response))):]
        echoed = match_loudness(fftconvolve(samples, response)[: len(samples)], samples)
        level = self.cfg.level.sample(rng)
        mixed = (1.0 - level) * samples + level * echoed
        return mixed, {"file": self.assets.name(path), "level": round(level, 4)}
