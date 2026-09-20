from typing import Any

from pydantic import PrivateAttr

from core import config as core_config
from core.config import PipelineConfig
from core.errors import ConfigError
from core.utils import langs
from core.utils.config import ConfigBody, as_component, require

DATASET_KEYS = {"dataset", "languages"}


class DatasetConfig(ConfigBody):
    name: str

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        name, spec = as_component(raw)
        return {"name": name, **spec}


class LanguagesConfig(ConfigBody):

    lang: str
    target: str

    @classmethod
    def expand(cls, raw: Any) -> Any:
        lang = _code(require(raw, "lang"), field="lang")
        target = _code(require(raw, "target"), field="target")
        if lang == target:
            raise ValueError("'lang' and 'target' must differ")
        return dict(lang=lang, target=target)

    def expected_target(self, src_lang: str) -> str:
        return self.lang if langs.norm_code(src_lang) == self.target else self.target


def _code(value: str, *, field: str) -> str:
    code = langs.norm_code(value)
    if not code:
        raise ValueError(f"unknown language {value!r} for {field!r} "
                         f"(known: {sorted(langs.CODE_TO_NAME)})")
    return code


class BenchConfig(ConfigBody):
    name: str
    dataset: DatasetConfig
    languages: LanguagesConfig
    stity: PipelineConfig

    _raw: dict = PrivateAttr(default_factory=dict)

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
        raise ConfigError(f"dataset config {dataset!r}: unknown key(s) {extra} "
                          f"(allowed: {sorted(DATASET_KEYS)})")
    spec = core_config.read_named(pipeline, "pipeline")
    raw = {"name": f"{pipeline}-{dataset}", **data, "stity": spec}
    cfg = BenchConfig.parse({**raw,
                             "stity": core_config.parse_pipeline(spec, name=pipeline)})
    cfg._raw = raw
    return cfg
