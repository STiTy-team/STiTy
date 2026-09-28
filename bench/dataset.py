import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from core.errors import DataError
from core.utils import langs
from core.utils.json import read_jsonl
from core.utils.audio import probe_duration

from .config import DatasetConfig

MANIFEST_NAME = "manifest.jsonl"
ALIGNMENT_NAME = "alignment.jsonl"
SPEC_NAME = "dataset.yml"


@dataclass(frozen=True)
class Item:
    id: str
    audio: Path
    duration_sec: float
    src_lang: str
    group: str
    speaker: str
    offset: float | None = None
    partial: bool = False
    reference: str = ""
    reference_translations: dict[str, str] = field(default_factory=dict)
    reference_alignment: tuple[dict, ...] = ()
    reference_segmentation: tuple[dict, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    spec: dict
    manifest_sha256: str
    alignment_sha256: str | None
    items: list[Item]

    @property
    def languages(self) -> list[str]:
        return [langs.norm_code(code) or code for code in self.spec["languages"]]

    @property
    def n_sessions(self) -> int:
        return len({i.group for i in self.items})

    def provenance(self) -> dict:
        return {
            **self.spec,
            "name": self.name,
            "root": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "alignment_sha256": self.alignment_sha256,
            "n_aligned": sum(1 for i in self.items if i.reference_alignment),
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
        duration_sec=float(raw["duration"]),
        src_lang=raw.get("src_lang") or default_lang,
        group=str(raw.get("group") or raw["id"]),
        speaker=str(raw.get("speaker") or ""),
        offset=float(raw["offset"]) if raw.get("offset") is not None else None,
        partial=bool(raw.get("partial")),
        reference=reference.get("transcript") or "",
        reference_translations=dict(reference.get("translations") or {}),
    )


def _read_alignment(path: Path, items: list[Item]) -> list[Item]:
    transcripts = {item.id: item.reference for item in items}
    words = {}
    for row in read_jsonl(path, strict=True):
        if transcripts.get(row["id"]) == row["transcript"]:
            words[row["id"]] = tuple(row["words"])
    return [replace(item, reference_alignment=words.get(item.id, ())) for item in items]


def _talks(items: list[Item]) -> list[Item]:
    by_group: dict[str, list[Item]] = {}
    for item in items:
        by_group.setdefault(item.group, []).append(item)

    talks = []
    for group, sentences in by_group.items():
        if len({s.audio for s in sentences}) != 1 or any(s.offset is None for s in sentences):
            raise DataError(
                f"group {group!r}: longform needs every item of a group to be an "
                f"offset window into one audio file"
            )
        sentences = sorted(sentences, key=lambda s: s.offset)
        targets = {lang for s in sentences for lang in s.reference_translations}
        talks.append(
            Item(
                id=group,
                audio=sentences[0].audio,
                duration_sec=probe_duration(sentences[0].audio),
                src_lang="+".join(dict.fromkeys(s.src_lang for s in sentences)),
                group=group,
                speaker=sentences[0].speaker,
                reference=" ".join(s.reference for s in sentences),
                reference_translations={
                    lang: " ".join(s.reference_translations.get(lang, "") for s in sentences)
                    for lang in targets
                },
                reference_alignment=tuple(
                    {**w, "start": w["start"] + s.offset, "end": w["end"] + s.offset}
                    for s in sentences
                    for w in s.reference_alignment
                ),
                reference_segmentation=tuple(
                    {
                        "offset": s.offset,
                        "duration": s.duration_sec,
                        "src_lang": s.src_lang,
                        "partial": s.partial,
                        "transcript": s.reference,
                        "translations": dict(s.reference_translations),
                    }
                    for s in sentences
                ),
            )
        )
    return talks


def load(cfg: DatasetConfig, root: Path) -> DatasetSpec:
    ds_root = root / cfg.name
    spec = _read_spec(ds_root)
    languages = spec.get("languages") or []
    if not languages:
        raise DataError(
            f"{ds_root / SPEC_NAME} has no 'languages'. It is what an item without "
            f"its own src_lang falls back to, so there is no default for it."
        )

    manifest_path = ds_root / str(spec.get("manifest") or MANIFEST_NAME)
    if not manifest_path.is_file():
        raise DataError(f"{manifest_path} is missing")
    items = sorted(
        (
            _parse_item(raw, root=ds_root, default_lang=languages[0])
            for raw in read_jsonl(manifest_path, strict=True)
        ),
        key=lambda item: item.group,
    )
    if not items:
        raise DataError(f"{manifest_path} has no items")

    alignment_path = ds_root / ALIGNMENT_NAME
    alignment_sha256 = None
    if alignment_path.is_file():
        alignment_sha256 = _sha256(alignment_path)
        items = _read_alignment(alignment_path, items)

    if cfg.longform:
        items = _talks(items)

    items = _pick(items, cfg)

    return DatasetSpec(
        name=str(spec.get("name") or cfg.name),
        root=ds_root,
        spec=spec,
        manifest_sha256=_sha256(manifest_path),
        alignment_sha256=alignment_sha256,
        items=items,
    )


def _pick(items: list[Item], cfg: DatasetConfig) -> list[Item]:
    if cfg.pick == "longest":
        chosen = {i.id for i in sorted(items, key=lambda i: -i.duration_sec)[: cfg.limit]}
        return [i for i in items if i.id in chosen]
    return items[: cfg.limit]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
