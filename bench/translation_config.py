from pathlib import Path
from typing import Any

import yaml
from pydantic import PrivateAttr, model_validator

from core.errors import ConfigError
from core.utils.config import ConfigBody, as_component

from .config import LanguagesConfig


class TranslationComponentConfig(ConfigBody):
    """A translation backend resolved without constructing an ASR pipeline."""

    name: str
    options: dict = {}

    @classmethod
    def normalize(cls, raw: Any) -> Any:
        from core.pipelines.translation import translators

        name, options = as_component(raw)
        validated = translators.get(name).validate(options, kind="translation")
        return {"name": name, "options": validated}


class RetranslateConfig(ConfigBody):
    """Config for replaying stored ASR commits through one translator."""

    name: str
    languages: LanguagesConfig
    translation: TranslationComponentConfig
    context_scope: str = "item"

    _raw: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_context_scope(self) -> "RetranslateConfig":
        if self.context_scope not in ("item", "group"):
            raise ValueError("context_scope must be 'item' or 'group'")
        return self

    @property
    def raw(self) -> dict:
        return self._raw


def load(path: str | Path) -> RetranslateConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RetranslateConfig.parse(raw)
    cfg._raw = raw if isinstance(raw, dict) else {}
    return cfg
