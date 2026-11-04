import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

from core.utils.config import ConfigBody

from ..assets import Assets
from ..config import PlaceConfig


class Effect(ABC):
    NAME: str

    def __init__(self, cfg: ConfigBody):
        self.cfg = cfg

    @classmethod
    def create(cls, cfg: ConfigBody, *, data_root: Path) -> Self:
        return cls(cfg)

    @property
    def fingerprint(self) -> str:
        return json.dumps(self.cfg.model_dump(), sort_keys=True)

    @abstractmethod
    def apply(self, samples: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]: ...


class PlaceEffect(Effect):
    cfg: PlaceConfig

    def __init__(self, cfg: PlaceConfig, assets: Assets):
        assets.check(cfg.place)
        super().__init__(cfg)
        self.assets = assets

    @classmethod
    def create(cls, cfg: PlaceConfig, *, data_root: Path) -> Self:
        return cls(cfg, Assets(data_root, cls.NAME))

    def pick_file(self, rng: np.random.Generator) -> Path:
        return pick(rng, self.assets.files(self.cfg.place))


def pick[T](rng: np.random.Generator, options: Sequence[T]) -> T:
    return options[int(rng.integers(len(options)))]


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0


def match_loudness(samples: np.ndarray, reference: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    own = rms(samples)
    return samples * (ratio * rms(reference) / own) if own > 0 else samples
