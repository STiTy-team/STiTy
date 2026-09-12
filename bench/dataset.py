import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from core.errors import DataError

from .config import DatasetConfig

MANIFEST_NAME = "manifest.jsonl"
SPEC_NAME = "dataset.yml"


@dataclass(frozen=True)
class Item:
    id: str
    audio: Path
    duration: float
    src_lang: str
    group: str
    speaker: str
    offset: float | None = None
    transcript: str = ""
    translations: dict[str, str] = field(default_factory=dict)

    def reference_translation(self, target_lang: str) -> str:
        return self.translations.get(target_lang, "")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    split: str
    languages: list[str]
    has_transcript: bool
    translation_langs: list[str]
    group_rule: str
    manifest_sha256: str
    items: list[Item]

    @property
    def n_sessions(self) -> int:
        return len({i.group for i in self.items})

    def provenance(self) -> dict:
        return {
            "name": self.name,
            "root": str(self.root),
            "split": self.split,
            "languages": list(self.languages),
            "provides": {"transcript": self.has_transcript,
                         "translations": list(self.translation_langs)},
            "group_rule": self.group_rule,
            "manifest_sha256": self.manifest_sha256,
            "n_items": len(self.items),
            "n_sessions": self.n_sessions,
        }


def _read_spec(root: Path) -> dict:
    path = root / SPEC_NAME
    if not path.is_file():
        raise DataError(
            f"{path} is missing. A bench dataset directory needs {SPEC_NAME} and "
            f"{MANIFEST_NAME}; see datasets/README.md for the contract."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_item(raw: dict, *, root: Path, default_lang: str) -> Item:
    reference = raw.get("reference") or {}
    return Item(
        id=raw["id"],
        audio=root / raw["audio"],
        duration=float(raw["duration"]),
        src_lang=raw.get("src_lang") or default_lang,
        group=str(raw.get("group") or raw["id"]),
        speaker=str(raw.get("speaker") or ""),
        offset=float(raw["offset"]) if raw.get("offset") is not None else None,
        transcript=reference.get("transcript") or "",
        translations=dict(reference.get("translations") or {}),
    )


def _iter_manifest(path: Path) -> Iterator[tuple[int, dict]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{lineno}: invalid JSON ({e})") from e


def _grouped_contiguously(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda item: item.group)


def load(cfg: DatasetConfig, root: Path) -> DatasetSpec:
    ds_root = root / cfg.name
    spec = _read_spec(ds_root)

    lang_codes = list(spec.get("languages") or [])
    provides = spec.get("provides") or {}
    translation_langs = list(provides.get("translations") or [])

    manifest_path = ds_root / str(spec.get("manifest") or MANIFEST_NAME)
    if not manifest_path.is_file():
        raise DataError(f"{manifest_path} is missing")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    if not lang_codes:
        raise DataError(
            f"{ds_root / SPEC_NAME} has no 'languages'. It is what an item without "
            f"its own src_lang falls back to, so there is no default for it."
        )
    default_lang = lang_codes[0]
    items = [_parse_item(raw, root=ds_root, default_lang=default_lang)
             for _, raw in _iter_manifest(manifest_path)]
    if not items:
        raise DataError(f"{manifest_path} has no items")

    items = _grouped_contiguously(items)
    if cfg.limit is not None:
        items = items[:cfg.limit]

    return DatasetSpec(
        name=str(spec.get("name") or cfg.name),
        root=ds_root,
        split=str(spec.get("split") or ""),
        languages=lang_codes,
        has_transcript=bool(provides.get("transcript", True)),
        translation_langs=translation_langs,
        group_rule=str(spec.get("group_rule") or "id"),
        manifest_sha256=digest,
        items=items,
    )
