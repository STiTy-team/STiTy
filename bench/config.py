import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import PrivateAttr

from core import config as core_config
from core.config import ConfigMeta, PipelineConfig
from core.errors import ConfigError
from core.utils import langs
from core.utils.config import ConfigBody, as_component
from core.utils.paths import get_project_root

from .augment import AugmentConfig

DATASET_KEYS = {"dataset", "target", "augment"}
NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9.+-]*(_[a-z0-9][a-z0-9.+-]*)*")


def get_runs_dir() -> Path:
    return get_project_root() / "bench" / "runs"


class DatasetConfig(ConfigBody):
    name: str
    longform: bool = False
    limit: int | None = None
    pick: Literal["first", "longest"] = "first"

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


def ref(name: str, version: int) -> str:
    return name if version == 1 else f"{name}@v{version}"


def config_hash(body: dict) -> str:
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def dataset_body(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in DATASET_KEYS}


def identity(name: str, meta: ConfigMeta, body: dict) -> dict:
    return {
        "name": name,
        "version": meta.version,
        "ref": ref(name, meta.version),
        "hash": config_hash(body),
        "description": meta.description,
        "tags": list(meta.tags),
    }


def check_name(name: str, kind: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise ConfigError(
            f"{kind} config name {name!r} does not follow the naming rule: lowercase "
            f"slots joined by '_', each made of [a-z0-9.+-] (see configs/README.md)"
        )


class BenchConfig(ConfigBody):
    name: str
    dataset: DatasetConfig
    target: str
    augment: AugmentConfig | None = None
    stity: PipelineConfig

    _raw: dict = PrivateAttr(default_factory=dict)
    _identity: dict = PrivateAttr(default_factory=dict)

    @classmethod
    def expand(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "target" in raw:
            return {**raw, "target": _code(raw["target"], field="target")}
        return raw

    @property
    def raw(self) -> dict:
        return self._raw

    @property
    def identity(self) -> dict:
        """`{"dataset": ..., "pipeline": ...}`, each as `identity()` builds it."""
        return self._identity

    @property
    def run_dir(self) -> Path:
        return get_runs_dir() / self.name

    def resolved(self) -> dict:
        return self.model_dump()


def parse(raw: dict) -> BenchConfig:
    cfg = BenchConfig.parse(raw)
    cfg._raw = raw if isinstance(raw, dict) else {}
    return cfg


def from_text(pipeline: str, pipeline_yaml: str, dataset: str, dataset_yaml: str) -> BenchConfig:
    check_name(dataset, "dataset")
    check_name(pipeline, "pipeline")
    return from_bodies(
        pipeline,
        core_config.split_meta(core_config.parse_yaml(pipeline_yaml, pipeline), pipeline),
        dataset,
        core_config.split_meta(core_config.parse_yaml(dataset_yaml, dataset), dataset),
    )


def from_bodies(
    pipeline: str,
    pipeline_with_meta: tuple[dict, ConfigMeta],
    dataset: str,
    dataset_with_meta: tuple[dict, ConfigMeta],
) -> BenchConfig:
    data, data_meta = dataset_with_meta
    spec, pipeline_meta = pipeline_with_meta
    check_dataset_keys(dataset, data)
    ids = {
        "dataset": identity(dataset, data_meta, data),
        "pipeline": identity(pipeline, pipeline_meta, spec),
    }
    raw = {"name": f"{ids['dataset']['ref']}/{ids['pipeline']['ref']}", **data, "stity": spec}
    cfg = BenchConfig.parse({**raw, "stity": core_config.parse_pipeline(spec, name=pipeline)})
    cfg._raw = raw
    cfg._identity = ids
    return cfg


def check_dataset_keys(name: str, data: dict) -> None:
    extra = sorted(set(data) - DATASET_KEYS)
    if extra:
        raise ConfigError(
            f"dataset config {name!r}: unknown key(s) {extra} "
            f"(allowed: {sorted(DATASET_KEYS | {core_config.META_KEY})})"
        )


def check_text(kind: str, name: str, text: str) -> dict:
    check_name(name, kind)
    body, meta = core_config.split_meta(core_config.parse_yaml(text, name), name)
    if kind == "pipeline":
        core_config.parse_pipeline(body, name=name)
    else:
        check_dataset_keys(name, body)
        DatasetConfig.parse(body.get("dataset"), root=f"dataset config {name!r} dataset")
        try:
            _code(str(body.get("target") or ""), field="target")
        except ValueError as e:
            raise ConfigError(f"dataset config {name!r}: {e}") from None
        if body.get("augment") is not None:
            AugmentConfig.parse(body["augment"], root=f"dataset config {name!r} augment")
    return identity(name, meta, body)
