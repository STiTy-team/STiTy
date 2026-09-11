from typing import Callable

from .errors import BenchConfigError

_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        table = _REGISTRY.setdefault(kind, {})
        if name in table:
            raise BenchConfigError(f"{kind} {name!r} is already registered by {table[name]!r}")
        table[name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    table = _REGISTRY.get(kind, {})
    if name not in table:
        raise BenchConfigError(
            f"unknown {kind}: {name!r} (available: {sorted(table) or 'none registered'})"
        )
    return table[name]
