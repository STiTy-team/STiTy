from typing import Any

import numpy as np
from pydantic import Field, model_serializer, model_validator

from core.utils.config import ConfigBody


class Range(ConfigBody):
    lo: float
    hi: float

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return {"lo": raw, "hi": raw}
        if isinstance(raw, list):
            if len(raw) != 2:
                raise ValueError(f"a range is written [low, high], got {raw}")
            return {"lo": raw[0], "hi": raw[1]}
        return raw

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.lo > self.hi:
            raise ValueError(f"need low <= high, got [{self.lo}, {self.hi}]")
        return self

    @model_serializer
    def _dump(self) -> list[float]:
        return [self.lo, self.hi]

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


class PlaceConfig(ConfigBody):
    place: str = Field(min_length=1)
    level: Range

    @model_validator(mode="after")
    def _not_negative(self) -> "PlaceConfig":
        if self.level.lo < 0:
            raise ValueError(f"level cannot be negative, got {self.level.lo}")
        return self


class NoiseConfig(PlaceConfig):
    pass


class RoomConfig(PlaceConfig):
    @model_validator(mode="after")
    def _at_most_one(self) -> "RoomConfig":
        if self.level.hi > 1:
            raise ValueError(f"room level is a share of the signal, at most 1, got {self.level.hi}")
        return self


class VolumeConfig(ConfigBody):
    min_level_db: Range

    @model_validator(mode="after")
    def _below_full_scale(self) -> "VolumeConfig":
        if self.min_level_db.hi > 0:
            raise ValueError(
                f"min_level_db is dB below full scale, at most 0, got {self.min_level_db.hi}"
            )
        return self


class AugmentConfig(ConfigBody):
    room: RoomConfig | None = None
    noise: NoiseConfig | None = None
    volume: VolumeConfig | None = None
