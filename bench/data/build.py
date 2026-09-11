"""Helper for dataset converters.

Converters live in datasets/<name>/convert.py and know the shape of one corpus.
This writes the parts every dataset shares, then validates, so no converter has to
remember the contract.
"""
import json
from pathlib import Path

import yaml

from . import audio as audio_mod
from . import manifest as manifest_mod


def target_root(name: str, explicit: str | None = None) -> Path:
    from .. import paths
    if explicit:
        return Path(explicit).expanduser()
    return paths.data_root(create=True) / name


def write_spec(root: Path, *, name: str, split: str, languages: list[str],
               audio_format: str, primary_metric: str, translations: list[str],
               group_rule: str, bench_defaults: dict,
               sample_rate: int = 16000) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": name,
        "split": split,
        "languages": list(languages),
        "audio": {"format": audio_format, "sample_rate": sample_rate},
        "provides": {"transcript": True, "translations": list(translations)},
        "primary_metric": primary_metric,
        "group_rule": group_rule,
        "bench_defaults": dict(bench_defaults),
    }
    path = root / "dataset.yml"
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def write_manifest(root: Path, rows) -> Path:
    path = root / "manifest.jsonl"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise SystemExit(f"no items written to {path} -- check the source paths")
    return path


def row(*, item_id: str, audio_rel: str, duration: float, src_lang: str,
        transcript: str, group: str | None = None, speaker: str = "",
        offset: float | None = None, translations: dict | None = None) -> dict:
    out = {
        "id": item_id,
        "audio": audio_rel,
        "duration": round(float(duration), 3),
        "group": group or item_id,
        "speaker": speaker,
        "src_lang": src_lang,
        "reference": {"transcript": transcript,
                      "translations": dict(translations or {})},
    }
    if offset is not None:
        out["offset"] = round(float(offset), 3)
    return out


def duration_of(path: Path, audio_format: str, sample_rate: int = 16000) -> float:
    return audio_mod.probe_duration(path, audio_format, sample_rate)


def finish(root: Path, *, name: str, check_audio: bool = True) -> int:
    """Loads what was just written and validates it. Converters end with this."""
    spec = manifest_mod.load(name, root=root)
    problems = manifest_mod.verify(spec, check_audio=check_audio)
    return manifest_mod.print_report(spec, problems, where=str(root))
