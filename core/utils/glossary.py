"""A shared list of names and terms, read by ASR biasing and by the translator.

One file per session or domain under `configs/glossaries/<name>.yml`. An entry is one
concept written in each language it is spoken or read in:

    terms:
      - {ko: 김태호, en: Kim Tae-ho}
      - {ko: 을지로, en: Euljiro}

ASR gets every spoken form; the translator gets the pairs whose source form occurs in
the line it is translating.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.errors import ConfigError
from core.utils import langs
from core.utils.paths import get_project_root


def glossary_path(name: str) -> Path:
    path = Path(name).expanduser()
    if path.suffix in (".yml", ".yaml") and path.is_file():
        return path
    directory = get_project_root() / "configs" / "glossaries"
    candidate = directory / f"{name}.yml"
    if candidate.is_file():
        return candidate
    available = sorted(p.stem for p in directory.glob("*.yml")) if directory.is_dir() else []
    raise ConfigError(f"unknown glossary {name!r} (available: {available or 'none'} in {directory})")


@dataclass(frozen=True)
class Glossary:
    name: str
    terms: tuple[dict[str, str], ...]

    @classmethod
    def load(cls, name: str) -> "Glossary":
        path = glossary_path(name)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        entries = raw.get("terms")
        if not isinstance(entries, list):
            raise ConfigError(f"{path}: 'terms' must be a list of {{language: form}} mappings")
        terms = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigError(f"{path}: terms[{index}] is not a mapping")
            forms = {}
            for code, form in entry.items():
                lang = langs.norm_code(str(code))
                if not lang:
                    raise ConfigError(f"{path}: terms[{index}] has unknown language {code!r}")
                if str(form).strip():
                    forms[lang] = str(form).strip()
            if forms:
                terms.append(forms)
        return cls(name=name, terms=tuple(terms))

    def spoken_forms(self, languages: list[str] | tuple = ()) -> list[str]:
        wanted = set(languages)
        forms = [form for entry in self.terms for lang, form in entry.items()
                 if not wanted or lang in wanted]
        return list(dict.fromkeys(forms))

    def pairs_in(self, text: str, source: str, target: str) -> list[tuple[str, str]]:
        lowered = (text or "").lower()
        return [(entry[source], entry[target]) for entry in self.terms
                if source in entry and target in entry and entry[source].lower() in lowered]

    def forms_in(self, text: str, language: str) -> list[dict[str, str]]:
        lowered = (text or "").lower()
        return [entry for entry in self.terms
                if language in entry and entry[language].lower() in lowered]
