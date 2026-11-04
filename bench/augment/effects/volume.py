import numpy as np

from ..config import VolumeConfig
from .base import Effect, rms


class Volume(Effect):
    NAME = "volume"
    cfg: VolumeConfig

    def apply(self, samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        min_level_db = self.cfg.min_level_db.sample(rng)
        loudness = rms(samples)
        target = 10 ** (min_level_db / 20)
        gain = 1.0
        if 0 < loudness < target:
            gain = min(target / loudness, 1.0 / float(np.max(np.abs(samples))))
        return samples * gain, {
            "min_level_db": round(min_level_db, 2),
            "gain_db": round(20 * float(np.log10(gain)), 2),
        }
