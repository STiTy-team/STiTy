from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from core.errors import DataError
from core.utils import audio

ASSETS_DIR = "augment"


class Assets:
    def __init__(self, data_root: Path, kind: str):
        self.data_root = data_root
        self.root = data_root / ASSETS_DIR / kind

    def places(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def files(self, place: str) -> tuple[Path, ...]:
        return _wavs(self.root / place)

    def check(self, place: str) -> None:
        if not self.files(place):
            raise DataError(
                f"no .wav files for {place!r} under {self.root} "
                f"(available: {self.places() or 'none'})"
            )

    def name(self, path: Path) -> str:
        return str(path.relative_to(self.data_root))


@cache
def _wavs(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(directory.rglob("*.wav")))


@lru_cache(maxsize=32)
def load(path: Path) -> np.ndarray:
    return audio.load_window(path)
