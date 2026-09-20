import importlib
import inspect
import pkgutil
from dataclasses import dataclass
from typing import Callable

from core.errors import ConfigError
from core.utils import timing


def discover(package: str, *, packages: bool = False) -> None:
    found = importlib.import_module(package)
    for module in pkgutil.iter_modules(found.__path__):
        if not module.name.startswith("_") and module.ispkg == packages:
            importlib.import_module(f"{package}.{module.name}")


@dataclass(frozen=True)
class Speech:

    started_at: float
    ended_at: float
    silence_waited_out_sec: float


@dataclass(frozen=True)
class Partial:

    text: str
    language: str
    seq: int


@dataclass(frozen=True)
class Transcribed:

    original: str
    language: str
    commit_reason: str
    decision_audio_sec: float
    recv_elapsed_sec: float


@dataclass(frozen=True)
class Final:

    original: str
    translation: str
    language: str
    target_lang: str
    commit_reason: str
    decision_audio_sec: float
    recv_elapsed_sec: float


class Component:

    SETTINGS: dict[str, tuple[str, Callable]] = {}

    def __init__(self, settings: dict, *, cfg):
        self.settings = settings
        self.cfg = cfg

    @classmethod
    def validate(cls, options: dict, *, kind: str) -> dict:
        unknown = sorted(set(options) - set(cls.SETTINGS))
        if unknown:
            raise ConfigError(
                f"stity.{kind}: {cls.NAME!r} has no setting(s) {unknown} "
                f"(it accepts: {sorted(cls.SETTINGS) or 'nothing'})"
            )
        return {stored: read(options[key])
                for key, (stored, read) in cls.SETTINGS.items()
                if options.get(key) is not None}

    async def load(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def start(self, **_) -> None:
        pass


LIFECYCLE = frozenset({"load", "close", "start", "validate"})


def _work_methods(cls: type) -> list[str]:
    for base in cls.__mro__:
        if Component in base.__bases__:
            return [name for name, value in vars(base).items()
                    if not name.startswith("_") and name not in LIFECYCLE
                    and inspect.isfunction(value)]
    return []


class Registry:

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            if name in self._entries:
                raise ConfigError(
                    f"cannot register {self.kind} {name!r}: "
                    f"already taken by {self._entries[name]!r}"
                )
            cls.NAME = name
            for method in _work_methods(cls):
                setattr(cls, method, timing.measure(method)(getattr(cls, method)))
            self._entries[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type:
        if name not in self._entries:
            raise ConfigError(
                f"unknown {self.kind}: {name!r} "
                f"(available: {self.names() or 'none registered'})"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        return f"({self.kind!r} Registry, [{self.names()}])"
