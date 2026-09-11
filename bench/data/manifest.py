import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import yaml

from .. import langs, paths
from ..errors import BenchDataError

AUDIO_FORMATS = {"flac", "wav", "pcm_s16le"}
PRIMARY_METRICS = {"wer", "cer"}

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
    audio_format: str
    sample_rate: int
    primary_metric: str
    has_transcript: bool
    translation_langs: list[str]
    group_rule: str
    bench_defaults: dict
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
            "audio_format": self.audio_format,
            "sample_rate": self.sample_rate,
            "primary_metric": self.primary_metric,
            "provides": {"transcript": self.has_transcript,
                         "translations": list(self.translation_langs)},
            "group_rule": self.group_rule,
            "manifest_sha256": self.manifest_sha256,
            "n_items": len(self.items),
            "n_sessions": self.n_sessions,
        }


def _spec_error(root: Path, msg: str) -> BenchDataError:
    return BenchDataError(f"{root / SPEC_NAME}: {msg}")


def _read_spec(root: Path) -> dict:
    path = root / SPEC_NAME
    if not path.is_file():
        raise BenchDataError(
            f"{path} is missing. A bench dataset directory needs {SPEC_NAME} and "
            f"{MANIFEST_NAME}; see datasets/README.md for the contract."
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise _spec_error(root, "expected a mapping")
    return raw


def _parse_item(raw: dict, *, root: Path, lineno: int, default_lang: str) -> Item:
    where = f"{root / MANIFEST_NAME}:{lineno}"
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise BenchDataError(f"{where}: 'id' is required")
    audio = raw.get("audio")
    if not isinstance(audio, str) or not audio:
        raise BenchDataError(f"{where}: 'audio' is required")
    if Path(audio).is_absolute():
        raise BenchDataError(
            f"{where}: 'audio' must be relative to the dataset directory, got {audio!r}. "
            f"Absolute paths break as soon as the data moves."
        )
    duration = raw.get("duration")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise BenchDataError(
            f"{where}: 'duration' is required and must be positive. It is LAAL's source "
            f"length T, so deriving it from the decoded audio would make LAAL depend on "
            f"the loader."
        )
    src_lang = raw.get("src_lang") or default_lang
    code = langs.norm_code(src_lang)
    if not code:
        raise BenchDataError(f"{where}: unknown src_lang {src_lang!r}")
    reference = raw.get("reference") or {}
    if not isinstance(reference, dict):
        raise BenchDataError(f"{where}: 'reference' must be a mapping")
    translations = reference.get("translations") or {}
    if not isinstance(translations, dict):
        raise BenchDataError(f"{where}: reference.translations must be a mapping")
    norm_translations = {}
    for lang, text in translations.items():
        lc = langs.norm_code(lang)
        if not lc:
            raise BenchDataError(f"{where}: unknown translation language {lang!r}")
        norm_translations[lc] = text or ""
    offset = raw.get("offset")
    if offset is not None and (not isinstance(offset, (int, float)) or offset < 0):
        raise BenchDataError(f"{where}: 'offset' must be a non-negative number")
    return Item(
        id=item_id,
        audio=root / audio,
        duration=float(duration),
        src_lang=code,
        group=str(raw.get("group") or item_id),
        speaker=str(raw.get("speaker") or ""),
        offset=float(offset) if offset is not None else None,
        transcript=(reference.get("transcript") or ""),
        translations=norm_translations,
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
                raise BenchDataError(f"{path}:{lineno}: invalid JSON ({e})") from e


def load(name: str, *, limit: int | None = None, root: Path | None = None) -> DatasetSpec:
    ds_root = Path(root) if root is not None else paths.dataset_dir(name)
    spec = _read_spec(ds_root)

    audio_spec = spec.get("audio") or {}
    if not isinstance(audio_spec, dict):
        raise _spec_error(ds_root, "'audio' must be a mapping")
    audio_format = audio_spec.get("format")
    if audio_format not in AUDIO_FORMATS:
        raise _spec_error(
            ds_root,
            f"audio.format must be one of {sorted(AUDIO_FORMATS)}, got {audio_format!r}. "
            f"Headerless PCM cannot be sniffed, so this has to be declared."
        )
    primary_metric = spec.get("primary_metric")
    if primary_metric not in PRIMARY_METRICS:
        raise _spec_error(
            ds_root, f"primary_metric must be one of {sorted(PRIMARY_METRICS)}, "
                     f"got {primary_metric!r}")

    declared = spec.get("languages") or []
    if not isinstance(declared, list) or not declared:
        raise _spec_error(ds_root, "'languages' must be a non-empty list of language codes")
    lang_codes = []
    for value in declared:
        code = langs.norm_code(value)
        if not code:
            raise _spec_error(ds_root, f"unknown language {value!r}")
        lang_codes.append(code)

    provides = spec.get("provides") or {}
    if not isinstance(provides, dict):
        raise _spec_error(ds_root, "'provides' must be a mapping")
    translation_langs = []
    for value in provides.get("translations") or []:
        code = langs.norm_code(value)
        if not code:
            raise _spec_error(ds_root, f"unknown translation language {value!r}")
        translation_langs.append(code)

    manifest_path = ds_root / str(spec.get("manifest") or MANIFEST_NAME)
    if not manifest_path.is_file():
        raise BenchDataError(f"{manifest_path} is missing")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    default_lang = lang_codes[0]
    items: list[Item] = []
    seen: set[str] = set()
    for lineno, raw in _iter_manifest(manifest_path):
        item = _parse_item(raw, root=ds_root, lineno=lineno, default_lang=default_lang)
        if item.id in seen:
            raise BenchDataError(f"{manifest_path}:{lineno}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise BenchDataError(f"{manifest_path} has no items")

    # Grouped contiguously so the handler lifetime derived from `group` is one
    # unbroken run per session. The sort is stable and keyed on `group` alone, so
    # the converter's ordering *within* a group survives -- ACL 60/60 needs that,
    # because gold sentence boundaries overlap and speaking order does not always
    # agree with seg-id order. Sorting by id here would silently reorder a talk.
    items.sort(key=lambda i: i.group)
    if limit is not None:
        items = items[:limit]

    return DatasetSpec(
        name=str(spec.get("name") or name),
        root=ds_root,
        split=str(spec.get("split") or ""),
        languages=lang_codes,
        audio_format=audio_format,
        sample_rate=int(audio_spec.get("sample_rate", 16000)),
        primary_metric=primary_metric,
        has_transcript=bool(provides.get("transcript", True)),
        translation_langs=translation_langs,
        group_rule=str(spec.get("group_rule") or "id"),
        bench_defaults=dict(spec.get("bench_defaults") or {}),
        manifest_sha256=digest,
        items=items,
    )


def verify(spec: DatasetSpec, *, check_audio: bool = True,
           duration_tolerance_sec: float = 0.05) -> list[str]:
    """Returns a list of problems; empty means the dataset is sound."""
    problems: list[str] = []
    for item in spec.items:
        if not item.audio.is_file():
            problems.append(f"{item.id}: audio missing at {item.audio}")
            continue
        if spec.has_transcript and not item.transcript.strip():
            problems.append(f"{item.id}: empty transcript")
        for lang in spec.translation_langs:
            if not item.translations.get(lang, "").strip():
                problems.append(f"{item.id}: missing {lang} reference translation")
        if not check_audio:
            continue
        try:
            from . import audio as audio_mod
            actual = audio_mod.probe_duration(item.audio, spec.audio_format, spec.sample_rate)
        except Exception as e:  # noqa: BLE001 - reported, not raised
            problems.append(f"{item.id}: could not read audio ({e})")
            continue
        if item.offset is None and abs(actual - item.duration) > duration_tolerance_sec:
            problems.append(
                f"{item.id}: duration {item.duration:.3f}s but the file is {actual:.3f}s")
    return problems


def print_report(spec: DatasetSpec, problems: list[str], *, where: str = "") -> int:
    """Prints the provenance block plus any problems, and returns an exit code."""
    print(json.dumps(spec.provenance(), indent=2, ensure_ascii=False))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems[:20]:
            print(f"  - {line}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        return 1
    print(f"\nOK: {len(spec.items)} items, {spec.n_sessions} sessions"
          + (f" -> {where}" if where else ""))
    return 0


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m bench.data.manifest")
    p.add_argument("--validate", metavar="DIR_OR_NAME", required=True,
                   help="dataset directory, or a name under $STITY_DATA_ROOT")
    p.add_argument("--no-audio", action="store_true",
                   help="skip decoding audio (path existence is still checked)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    candidate = Path(args.validate)
    try:
        if candidate.is_dir():
            spec = load(candidate.name, limit=args.limit, root=candidate)
        else:
            spec = load(args.validate, limit=args.limit)
    except BenchDataError as e:
        print(f"invalid: {e}")
        return 2

    return print_report(spec, verify(spec, check_audio=not args.no_audio))


if __name__ == "__main__":
    raise SystemExit(_main())
