from dataclasses import dataclass
from typing import Callable

from core.errors import ConfigError


@dataclass(frozen=True)
class Speech:

    started_at: float
    ended_at: float
    silence_waited_out_sec: float


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
