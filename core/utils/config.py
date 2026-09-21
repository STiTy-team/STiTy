"""Config bodies: one class per block of a config file, nested into a tree.

A body declares its fields. Nesting is by annotation -- a field typed as another
body -- so the tree parses itself and every error comes back with the path it was
found at, however deep. Nothing walks the tree by hand.

Three hooks cover the ways a block as written differs from the block as stored.
Most bodies need none of them:

    normalize   a shorthand to a mapping      `vad: true` -> `{enabled: true}`
    wire_keys   what the file may contain     defaults to the fields
    expand      written form to stored form   `commit: seg` -> the seven flags

Messages raised inside a body never name their own path. The path comes from where
the body sits in the tree, so the same body used in two places reports correctly in
both. `format_errors` puts the two together.

The blocks, and the rules inside them, belong to whatever is being configured. This
module is only the machinery.
"""
from pathlib import Path
from typing import Any, TypeVar, get_args

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from core.errors import ConfigError


B = TypeVar("B", bound="ConfigBody")


def reject_unknown(mapping: dict, allowed: set[str]) -> None:
    """Naming what was allowed is most of the value of catching a typo.

    pydantic's own `extra='forbid'` reports the offending key but not the
    alternatives, which is the half the reader actually needs.
    """
    extra = sorted(set(mapping) - set(allowed))
    if extra:
        raise ValueError(f"unknown key(s) {extra} (allowed: {sorted(allowed)})")


def require(mapping: dict, key: str):
    """A key that has no default and no sensible absence."""
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{key!r} is required")
    return mapping[key]


def as_component(raw: Any) -> tuple[str, dict]:
    """`google` and `{name: google, ...}` are one thing written two ways.

    Returns the name and whatever else was written alongside it.
    """
    if isinstance(raw, str):
        return raw, {}
    if isinstance(raw, dict):
        spec = dict(raw)
        name = spec.pop("name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("mapping form needs a 'name' key")
        return name, spec
    raise ValueError(f"expected a name or a mapping, got {type(raw).__name__}")


class ConfigBody(BaseModel):
    """One block of a config file.

    Unknown keys are an error rather than a shrug: a setting that is silently
    dropped still lets the run finish and produce a plausible number, which is
    worse than not running at all. Bodies are frozen, so what was parsed is what
    everything downstream sees.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        """A shorthand form to the mapping it stands for. Override to accept one."""
        return raw

    @classmethod
    def wire_keys(cls) -> set[str]:
        """Keys the file may contain. Override when a body stores fields it does not
        accept (derived ones) or accepts keys it does not store."""
        return set(cls.model_fields)

    @classmethod
    def expand(cls, raw: Any) -> Any:
        """The mapping as written to the field values. Override to derive fields."""
        return raw

    @model_validator(mode="before")
    @classmethod
    def from_wire(cls, raw: Any) -> Any:
        raw = cls.normalize(raw)
        if isinstance(raw, dict) and cls.rewrites_keys():
            # A body that rewrites its keys has to be checked before it does so --
            # after expand() the written keys are gone. Everything else is left to
            # extra='forbid', which reports alongside the rest instead of aborting
            # the whole tree on the first typo.
            reject_unknown(raw, cls.wire_keys())
        return cls.expand(raw)

    @classmethod
    def rewrites_keys(cls) -> bool:
        """Whether the keys in the file differ from the fields on the body.

        `__func__` because comparing classmethods directly compares freshly bound
        objects, which are never identical even when nothing was overridden.
        """
        return (cls.expand.__func__ is not ConfigBody.expand.__func__
                or cls.wire_keys() != set(cls.model_fields))

    @classmethod
    def parse(cls: type[B], raw: Any, *, root: str = "config root") -> B:
        """Build from an already-loaded mapping. Every problem is reported, not just
        the first, so a config with five mistakes takes one round trip."""
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(format_errors(exc, model=cls, root=root)) from None

    @classmethod
    def load(cls: type[B], path: str | Path, *, root: str = "config root") -> B:
        """Read a YAML file into this body."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with open(p, encoding="utf-8") as f:
            return cls.parse(yaml.safe_load(f), root=root)


def body_at(model: type["ConfigBody"] | None, loc: tuple) -> type["ConfigBody"] | None:
    """The body that owns a location in the tree, found by walking the annotations."""
    for part in loc:
        fields = getattr(model, "model_fields", None)
        if fields is None or str(part) not in fields:
            return None
        model = body_type(fields[str(part)].annotation)
        if model is None:
            return None
    return model


def body_type(annotation: Any) -> type["ConfigBody"] | None:
    """The ConfigBody in an annotation, looking inside `X | None` and friends."""
    if isinstance(annotation, type) and issubclass(annotation, ConfigBody):
        return annotation
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, ConfigBody):
            return arg
    return None


def format_errors(exc: ValidationError, *, model: type[ConfigBody] | None = None,
                  root: str = "config root") -> str:
    """One line per problem, each prefixed with where it was found.

    Unknown keys are grouped by the block they appeared in, so one mistyped block
    does not become one line per key, and `model` lets each group say what the block
    would have accepted instead -- which is most of the value of catching a typo.
    """
    lines: list[str] = []
    extras: dict[tuple, list[str]] = {}

    for err in exc.errors():
        loc = err["loc"]
        where = ".".join(str(part) for part in loc) or root
        parent = ".".join(str(part) for part in loc[:-1]) or root
        if err["type"] == "extra_forbidden":
            extras.setdefault(tuple(loc[:-1]), []).append(str(loc[-1]))
            continue
        if err["type"] == "missing":
            lines.append(f"{parent}: {str(loc[-1])!r} is required")
            continue
        lines.append(f"{where}: {err['msg'].removeprefix('Value error, ')}")

    for loc, keys in extras.items():
        where = ".".join(str(part) for part in loc) or root
        body = body_at(model, loc)
        allowed = f" (allowed: {sorted(body.wire_keys())})" if body else ""
        lines.append(f"{where}: unknown key(s) {sorted(keys)}{allowed}")

    seen = list(dict.fromkeys(lines))
    if len(seen) == 1:
        return seen[0]
    return (f"{len(seen)} problems in the config:\n"
            + "\n".join(f"  - {line}" for line in seen))
