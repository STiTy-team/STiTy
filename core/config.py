"""The pipeline config: the system under test, shared by every entry point.

A pipeline config names the parts and how they commit. It says nothing about what
feeds them -- audio from a dataset, audio from a socket -- so bench and the server
read the same file and each brings its own other half.

Configs are addressed by name. `configs/pipelines/baseline.yml` is `baseline`, and
a name that is not there reports the ones that are.
"""
from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator

from core.errors import ConfigError
from core.utils.config import ConfigBody, as_component

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"

COMMIT_MODES = {
    "seg":    dict(always_commit=False, enable_dot_commit=False, hide_seg=False),
    "punct":  dict(always_commit=False, enable_dot_commit=True, hide_seg=True),
    "always": dict(always_commit=True, enable_dot_commit=False, hide_seg=True),
}


def resolve(name: str, kind: str) -> Path:
    """A bare name to the file it stands for, under `configs/<kind>s/`."""
    directory = CONFIG_ROOT / f"{kind}s"
    path = directory / f"{name}.yml"
    if path.is_file():
        return path
    available = sorted(p.stem for p in directory.glob("*.yml"))
    raise ConfigError(
        f"unknown {kind} config {name!r} "
        f"(available: {available or 'none'} in {directory})"
    )


def read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        found = "nothing" if raw is None else type(raw).__name__
        raise ConfigError(f"{path} must hold a mapping, found {found}")
    return raw


def read_named(name: str, kind: str) -> dict:
    return read(resolve(name, kind))


class ComponentConfig(ConfigBody):

    name: str
    options: dict = {}


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


class PipelineConfig(ConfigBody):
    """One `configs/pipelines/*.yml` file, with its parts already resolved."""

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

    @model_validator(mode="after")
    def resolve_parts(self) -> "PipelineConfig":
        from core import pipeline

        object.__setattr__(self, "resolved", pipeline.validate(self))
        return self


def parse_pipeline(raw: dict, *, name: str) -> PipelineConfig:
    return PipelineConfig.parse(raw, root=f"pipeline config {name!r}")


def load_pipeline(name: str) -> PipelineConfig:
    return parse_pipeline(read_named(name, "pipeline"), name=name)
