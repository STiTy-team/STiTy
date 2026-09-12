from pathlib import Path
from typing import Any

import yaml
from pydantic import PrivateAttr, model_validator

from core.errors import ConfigError
from core.utils import langs
from core.utils.config import ConfigBody, as_component, require

COMMIT_MODES = {
    "seg":    dict(always_commit=False, enable_dot_commit=False, hide_seg=False),
    "punct":  dict(always_commit=False, enable_dot_commit=True, hide_seg=True),
    "always": dict(always_commit=True, enable_dot_commit=False, hide_seg=True),
}


class ComponentConfig(ConfigBody):

    name: str
    options: dict = {}


class DatasetConfig(ConfigBody):
    name: str
    limit: int | None = None

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        name, spec = as_component(raw)
        return {"name": name, **spec}


class LanguagesConfig(ConfigBody):

    lang: str
    target: str
    restrict: bool = True

    @classmethod
    def expand(cls, raw: Any) -> Any:
        lang = _code(require(raw, "lang"), field="lang")
        target = _code(require(raw, "target"), field="target")
        if lang == target:
            raise ValueError("'lang' and 'target' must differ")
        return dict(lang=lang, target=target, restrict=bool(raw.get("restrict", True)))

    def expected_target(self, src_lang: str) -> str:
        return self.lang if langs.norm_code(src_lang) == self.target else self.target


def _code(value: str, *, field: str) -> str:
    code = langs.norm_code(value)
    if not code:
        raise ValueError(f"unknown language {value!r} for {field!r} "
                         f"(known: {sorted(langs.CODE_TO_NAME)})")
    return code


class CommitConfig(ConfigBody):
    mode: str
    always_commit: bool
    enable_dot_commit: bool
    hide_seg: bool

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        name, spec = as_component(raw)
        return {"name": name, **spec}

    @classmethod
    def wire_keys(cls) -> set[str]:
        return {"name", "hide_seg"}

    @classmethod
    def expand(cls, raw: Any) -> Any:
        mode = raw["name"]
        if mode not in COMMIT_MODES:
            raise ValueError(f"unknown mode {mode!r} (available: {sorted(COMMIT_MODES)})")
        table = COMMIT_MODES[mode]
        return dict(mode=mode, **{**table,
                                  "hide_seg": raw.get("hide_seg", table["hide_seg"])})


class StityConfig(ConfigBody):

    pipeline: ComponentConfig
    commit: CommitConfig
    gpu_memory_utilization: float
    resolved: dict = {}

    @classmethod
    def wire_keys(cls) -> set[str]:
        return {"pipeline", "commit", "gpu_memory_utilization"}

    @classmethod
    def expand(cls, raw: Any) -> Any:
        name, options = as_component(raw.get("pipeline") or "cascade")
        if raw.get("gpu_memory_utilization") is None:
            raise ValueError(
                "gpu_memory_utilization is required and has no default. vLLM's own "
                "default (0.8) means 'reserve everything spare', which has killed a "
                "co-tenant job on this machine. Use 0.5 when sharing the card."
            )
        return dict(pipeline={"name": name, "options": options},
                    commit=raw.get("commit"),
                    gpu_memory_utilization=raw["gpu_memory_utilization"])

    def part(self, kind: str) -> dict:
        return (self.resolved.get("parts") or {}).get(kind, {}).get("kwargs", {})


class PacingConfig(ConfigBody):

    chunk_size_ms: int = 200
    trailing_silence_ms: int = 1000
    realtime: bool = True


class BenchConfig(ConfigBody):
    name: str
    dataset: DatasetConfig
    languages: LanguagesConfig
    stity: StityConfig
    pacing: PacingConfig = PacingConfig()

    _raw: dict = PrivateAttr(default_factory=dict)

    @property
    def raw(self) -> dict:
        return self._raw

    @model_validator(mode="after")
    def resolve_pipeline(self) -> "BenchConfig":
        from core import pipelines

        object.__setattr__(self, "stity", self.stity.model_copy(
            update={"resolved": pipelines.validate(self.stity)}))
        return self

    def resolved(self) -> dict:
        return self.model_dump()


def parse(raw: dict) -> BenchConfig:
    cfg = BenchConfig.parse(raw)
    cfg._raw = raw if isinstance(raw, dict) else {}
    return cfg


def load(path: str | Path) -> BenchConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return parse(yaml.safe_load(f))
