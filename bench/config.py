from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from core import config as core_config
from core.config import PipelineConfig
from core.errors import ConfigError
from core.utils import env, langs
from core.utils.config import ConfigBody, as_component
from core.utils.paths import get_project_root

from .augment import AugmentConfig

DATASET_KEYS = {"dataset", "target", "augment"}
DATA_ROOT_ENV = "STITY_DATA_ROOT"


def get_runs_dir() -> Path:
    return get_project_root() / "bench" / "runs"


def get_data_root() -> Path:
    root = env.path(DATA_ROOT_ENV)
    if root is None:
        raise ConfigError(
            f"{DATA_ROOT_ENV} is not set. Point it at the directory holding the "
            f"converted datasets, e.g. export {DATA_ROOT_ENV}=~/datasets"
        )
    return root


class DatasetConfig(ConfigBody):
    name: str
    longform: bool = False
    limit: int | None = None

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        name, spec = as_component(raw)
        return {"name": name, **spec}


def _code(value: str, *, field: str) -> str:
    code = langs.norm_code(value)
    if not code:
        raise ValueError(
            f"unknown language {value!r} for {field!r} (known: {sorted(langs.CODE_TO_NAME)})"
        )
    return code


class BenchConfig(ConfigBody):
    name: str
    dataset: DatasetConfig
    target: str
    augment: AugmentConfig | None = None
    stity: PipelineConfig

    _raw: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def expand(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "target" in raw:
            return {**raw, "target": _code(raw["target"], field="target")}
        return raw

    @property
    def raw(self) -> dict:
        return self._raw

    def resolved(self) -> dict:
        return self.model_dump()


def parse(raw: dict) -> BenchConfig:
    cfg = BenchConfig.parse(raw)
    cfg._raw = raw if isinstance(raw, dict) else {}
    return cfg


def load(pipeline: str, dataset: str) -> BenchConfig:
    """One run is one pipeline fed by one dataset, named on the command line."""
    data = core_config.read_named(dataset, "dataset")
    extra = sorted(set(data) - DATASET_KEYS)
    if extra:
        raise ConfigError(
            f"dataset config {dataset!r}: unknown key(s) {extra} (allowed: {sorted(DATASET_KEYS)})"
        )
    spec = core_config.read_named(pipeline, "pipeline")
    raw = {"name": f"{pipeline}-{dataset}", **data, "stity": spec}
    cfg = BenchConfig.parse({**raw, "stity": core_config.parse_pipeline(spec, name=pipeline)})
    cfg._raw = raw
    return cfg
