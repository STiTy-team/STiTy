"""Browsing and recording into the datasets repo (`STITY_DATA_ROOT`), ported from the
standalone `datasets/manager/` tool so the same two jobs -- what is in a dataset, and
recording new items straight into it -- live in the one manager the team already has open.

`manifest.jsonl` is the live thing here: it is read and rewritten in place. A
`convert.py` re-run regenerates it and discards edits made here, which is fine for the
converted datasets and moot for a hand-recorded one.

The dataset repo is deliberately self-contained (see its own `_contract.py`), so this
module reaches into it by path rather than importing it as a package.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import statistics
import sys
import wave
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import ModuleType

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Under this and it was a misfire -- a double-tapped space bar, a click on the wrong
# button. Keeping them means hand-deleting them later.
MIN_DURATION_SEC = 0.25

# One take held in memory before it is posted. 16 kHz mono s16le is about 1.9 MB a
# minute, so this is roughly nine hours: a ceiling against a runaway client, not
# against any real session.
MAX_RECORD_BYTES = 1 << 30

PAGE = 100
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SAFE_DATASET = re.compile(r"^[A-Za-z0-9_-]{1,64}(/[A-Za-z0-9_.+-]{1,128})?$")
SAFE_LANG = re.compile(r"^[A-Za-z0-9_-]{1,16}$")
MARKERS = ("dataset.yml", "manifest.jsonl")
DURATION_BINS = (1, 2, 5, 10, 20, 30, 60, 180, 600)
SKIP = {"manager", "data", "audio", "__pycache__"}

README_STUB = """# {name}

데이터셋 매니저에서 만든 데이터셋. 받아오는 코퍼스가 아니라 직접 녹음해서 채운다.

`dataset.yml` 과 `manifest.jsonl` 은 매니저가 읽고 쓴다. 형식과 녹음하는 법은
`datasets/manager/README.md` 에 있다.
"""


class DatasetError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


_contract_cache: dict[str, ModuleType] = {}


def _contract(root: Path) -> ModuleType:
    """The dataset repo's own `_contract.py`, imported from `root` (= `STITY_DATA_ROOT`).

    That repo deliberately does not import STiTy, so this is the reverse direction:
    STiTy reaches into it, the same way `datasets/manager/serve.py` already does.
    """
    key = str(root)
    module = _contract_cache.get(key)
    if module is None:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import _contract as module  # type: ignore[import-not-found]

        _contract_cache[key] = module
    return module


def _plain_dirs(path: Path) -> list[Path]:
    return sorted(d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name not in SKIP)


def _is_dataset(path: Path) -> bool:
    return any((path / m).is_file() for m in MARKERS)


def list_dataset_dirs(root: Path) -> dict[str, Path]:
    """Every dataset under the repo root, keyed by its bench name.

    A multilingual corpus keeps one dataset per source language in a subdirectory
    (`fleurs/en_us`), so one level down counts too. A top-level directory is listed
    itself when it holds a dataset or holds none below it: requiring a dataset.yml
    would hide a directory you just made to record into, which is the one case where
    you most want to see it listed.
    """
    out = {}
    for top in _plain_dirs(root):
        nested = [d for d in _plain_dirs(top) if _is_dataset(d)]
        if _is_dataset(top) or not nested:
            out[top.name] = top
        for sub in nested:
            out[f"{top.name}/{sub.name}"] = sub
    return out


def read_spec(dataset: Path) -> dict:
    path = dataset / "dataset.yml"
    if not path.is_file():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_spec_raw(dataset: Path, spec: dict) -> None:
    """Rewrite dataset.yml from a loaded spec, keeping the contract's key order."""
    import yaml

    (dataset / "dataset.yml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")


def parse_langs(raw: str) -> list[str]:
    """"ko, en de" -> ["ko", "en", "de"], order kept, duplicates dropped."""
    out = []
    for lang in re.split(r"[,\s]+", raw.strip()):
        if not lang:
            continue
        if not SAFE_LANG.match(lang):
            raise DatasetError(400, f"언어 코드가 잘못됐다: {lang}")
        if lang not in out:
            out.append(lang)
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_manifest(dataset: Path) -> list[dict]:
    return read_jsonl(dataset / "manifest.jsonl")


def write_manifest(dataset: Path, rows: list[dict]) -> None:
    """Rewrite manifest.jsonl. Unlike the contract's writer, an empty dataset is
    allowed: deleting the last item is a legitimate thing to do from here."""
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def audio_path(dataset: Path, item: dict) -> Path | None:
    """The item's audio file, or None if the row points outside the dataset.

    Checked lexically, not on the resolved path: `audio` is normally a symlink to the
    corpus somewhere else on disk, which is inside the dataset as far as the row is
    concerned.
    """
    rel = str(item.get("audio") or "")
    if not rel or Path(rel).is_absolute():
        return None
    path = Path(os.path.normpath(dataset / rel))
    if not path.is_relative_to(dataset):
        return None
    return path


def text_units(text: str, lang: str, contract: ModuleType) -> int:
    if lang in contract.UNSPACED_LANGUAGES:
        return len(re.sub(r"\s", "", text))
    return len(text.split())


def spread(values: list[float]) -> dict | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": values[0],
        "p50": statistics.median(values),
        "p90": values[int(0.9 * (len(values) - 1))],
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def histogram(durations: list[float]) -> list[dict]:
    edges = [0, *DURATION_BINS, float("inf")]
    counts = [0] * (len(edges) - 1)
    for d in durations:
        for i in range(len(counts)):
            if edges[i] <= d < edges[i + 1]:
                counts[i] += 1
                break
    return [
        {"lo": edges[i], "hi": None if edges[i + 1] == float("inf") else edges[i + 1], "count": c}
        for i, c in enumerate(counts)
    ]


def fields(rows: list[dict]) -> list[dict]:
    """Which keys the rows carry, how often, and as what JSON type."""
    seen: dict[str, Counter] = {}

    def note(key: str, value) -> None:
        seen.setdefault(key, Counter())[type(value).__name__] += 1

    for row in rows:
        for key, value in row.items():
            note(key, value)
            if key == "reference" and isinstance(value, dict):
                for sub, inner in value.items():
                    note(f"reference.{sub}", inner)
                    if sub == "translations" and isinstance(inner, dict):
                        for lang, text in inner.items():
                            note(f"reference.translations.{lang}", text)
    return [{"key": k, "count": sum(c.values()), "types": dict(c)} for k, c in seen.items()]


def matches(row: dict, needle: str) -> bool:
    ref = row.get("reference") or {}
    haystack = [
        row.get("id", ""),
        row.get("speaker", ""),
        row.get("group", ""),
        ref.get("transcript") or "",
        *(ref.get("translations") or {}).values(),
    ]
    return any(needle in str(h).lower() for h in haystack)


class DatasetManager:
    def __init__(self, root: Path):
        self.root = root

    def contract(self) -> ModuleType:
        return _contract(self.root)

    def pick(self, name: str) -> Path:
        if not SAFE_DATASET.match(name or ""):
            raise DatasetError(400, "데이터셋 이름이 잘못됐다")
        dataset = list_dataset_dirs(self.root).get(name)
        if not dataset:
            raise DatasetError(404, f"그런 데이터셋이 없다: {name}")
        return dataset

    def find(self, dataset: Path, item_id: str) -> tuple[list[dict], int]:
        rows = read_manifest(dataset)
        for i, r in enumerate(rows):
            if r.get("id") == item_id:
                return rows, i
        raise DatasetError(404, f"그런 항목이 없다: {item_id}")

    # ---- endpoints ----

    def list_datasets(self) -> dict:
        out = []
        for name, dataset in list_dataset_dirs(self.root).items():
            rows = read_manifest(dataset)
            out.append(
                {
                    "name": name,
                    "items": len(rows),
                    "duration": round(sum(float(r.get("duration") or 0) for r in rows), 1),
                    "converted": (dataset / "manifest.jsonl").is_file(),
                }
            )
        return {"root": str(self.root), "datasets": out}

    def create_dataset(self, name: str, languages_raw: str) -> dict:
        """Make a new dataset directory at the repo root.

        The spec is written now rather than left to the first take, so the dataset has
        a declared src_lang from the moment it exists.
        """
        name = (name or "").strip()
        if not SAFE_NAME.match(name):
            raise DatasetError(400, "이름은 영문·숫자·_·- 만, 64자 이내")
        if name in SKIP:
            raise DatasetError(400, f"예약된 이름이다: {name}")
        dataset = self.root / name
        if dataset.exists():
            raise DatasetError(409, f"이미 있다: {name}")

        languages = parse_langs(languages_raw) or ["en"]
        dataset.mkdir()
        self.contract().write_spec(
            dataset,
            name=name,
            split="record",
            languages=languages,
            primary_metric="wer",
            translations=[],
            transcript=False,
            group_rule="id",
            bench_defaults={},
        )
        (dataset / "README.md").write_text(README_STUB.format(name=name), encoding="utf-8")
        return {"name": name, "path": str(dataset), "languages": languages}

    def shape(self, dataset: Path) -> dict:
        contract = self.contract()
        rows = read_manifest(dataset)
        spec = read_spec(dataset)
        manifest = dataset / "manifest.jsonl"
        aligned = contract._read_alignment(dataset / contract.ALIGNMENT_NAME)

        durations = [float(r.get("duration") or 0) for r in rows]
        files = {audio_path(dataset, r) for r in rows}
        missing = sum(1 for f in files if not f or not f.is_file())

        units = [
            text_units(t, r.get("src_lang", ""), contract)
            for r in rows
            if (t := (r.get("reference") or {}).get("transcript") or "").strip()
        ]
        translation_langs = Counter(
            lang
            for r in rows
            for lang, text in ((r.get("reference") or {}).get("translations") or {}).items()
            if (text or "").strip()
        )

        transcripts = {r.get("id"): (r.get("reference") or {}).get("transcript") or "" for r in rows}
        orphan = sum(1 for i in aligned if i not in transcripts)
        stale = sum(1 for i, a in aligned.items() if i in transcripts and a.get("transcript") != transcripts[i])

        groups = [r.get("group", r.get("id")) for r in rows]
        return {
            "spec": spec,
            "files": {
                "manifest": manifest.is_file(),
                "manifest_bytes": manifest.stat().st_size if manifest.is_file() else 0,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
                "alignment": (dataset / contract.ALIGNMENT_NAME).is_file(),
                "readme": (dataset / "README.md").is_file(),
                "convert": (dataset / "convert.py").is_file(),
            },
            "items": len(rows),
            "duration": {"total": sum(durations), **(spread(durations) or {})},
            "histogram": histogram(durations),
            "sessions": len(dict.fromkeys(groups)),
            "sessions_contiguous": len(dict.fromkeys(groups))
            == len([g for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g]),
            "speakers": len({r.get("speaker") for r in rows if r.get("speaker")}),
            "src_lang": dict(Counter(r.get("src_lang", "") for r in rows).most_common()),
            "audio_files": len(files),
            "audio_missing": missing,
            "offset_items": sum(1 for r in rows if r.get("offset") is not None),
            "partial_items": sum(1 for r in rows if r.get("partial")),
            "transcript": {"filled": len(units), "units": spread(units)},
            "translations": dict(translation_langs.most_common()),
            "fields": fields(rows),
            "alignment": {
                "items": len(aligned) - orphan,
                "stale": stale,
                "orphan": orphan,
                "aligners": dict(Counter(a.get("aligner", "") for a in aligned.values())),
            },
        }

    def items(self, dataset: Path, *, offset: int, q: str, order: str) -> dict:
        needle = (q or "").strip().lower()
        rows = read_manifest(dataset)
        indexed = [(i, r) for i, r in enumerate(rows) if not needle or matches(r, needle)]
        if order == "recent":
            indexed.reverse()
        page = []
        for i, row in indexed[offset : offset + PAGE]:
            audio = audio_path(dataset, row)
            page.append({**row, "line": i + 1, "has_audio": bool(audio and audio.is_file())})
        return {"total": len(rows), "matched": len(indexed), "offset": offset, "items": page}

    def item(self, dataset: Path, item_id: str) -> dict:
        contract = self.contract()
        rows, i = self.find(dataset, item_id)
        row = rows[i]
        path = audio_path(dataset, row)
        audio = None
        if path and path.is_file():
            import soundfile as sf

            info = sf.info(str(path))
            audio = {
                "file": str(Path(row["audio"])),
                "resolved": str(path),
                "bytes": path.stat().st_size,
                "format": info.format,
                "subtype": info.subtype,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_duration": info.frames / info.samplerate,
                "problem": contract.audio_problem(path),
            }
        aligned = contract._read_alignment(dataset / contract.ALIGNMENT_NAME).get(row["id"])
        if aligned:
            aligned = {**aligned, "stale": aligned.get("transcript") != (row.get("reference") or {}).get("transcript")}
        return {"line": i + 1, "row": row, "audio": audio, "alignment": aligned}

    def audio_bytes(self, dataset: Path, item_id: str) -> bytes:
        """The item's audio. An item cut from a longer file by `offset` gets only its
        window, so playback and word timings line up with what bench reads."""
        rows, i = self.find(dataset, item_id)
        row = rows[i]
        path = audio_path(dataset, row)
        if not path or not path.is_file():
            raise DatasetError(404, "오디오가 없다")
        if row.get("offset") is None:
            return path.read_bytes()
        import soundfile as sf

        with sf.SoundFile(str(path)) as f:
            f.seek(round(float(row["offset"]) * f.samplerate))
            frames = f.read(round(float(row["duration"]) * f.samplerate), dtype="int16")
            rate = f.samplerate
        buf = io.BytesIO()
        sf.write(buf, frames, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def verify(self, dataset: Path) -> dict:
        if not all((dataset / m).is_file() for m in MARKERS):
            raise DatasetError(409, "dataset.yml 과 manifest.jsonl 이 둘 다 있어야 검증한다")
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.contract().verify(dataset)
        return {"ok": code == 0, "report": out.getvalue().strip()}

    def drop_audio(self, dataset: Path, item_id: str) -> dict:
        """Unlink the file, keep the row. verify() will then flag it as missing, which
        is the honest state: the item exists and its audio does not. Items sharing one
        file by offset lose it together, so that is refused."""
        rows, i = self.find(dataset, item_id)
        path = audio_path(dataset, rows[i])
        if path and sum(audio_path(dataset, r) == path for r in rows) > 1:
            raise DatasetError(409, "다른 항목도 이 파일을 쓴다")
        if path:
            path.unlink(missing_ok=True)
        return {"deleted_audio": item_id}

    def drop_item(self, dataset: Path, item_id: str) -> dict:
        rows, i = self.find(dataset, item_id)
        path = audio_path(dataset, rows[i])
        shared = path and sum(audio_path(dataset, r) == path for r in rows) > 1
        if path and not shared:
            path.unlink(missing_ok=True)
        del rows[i]
        write_manifest(dataset, rows)
        return {"deleted": item_id}

    def record(
        self, dataset: Path, pcm: bytes, *, prefix: str, speakers: int, src_lang: str, speaker: str
    ) -> dict:
        """Raw s16le bytes in, a manifest row out.

        The page is the only client, so there is no reason to wrap one blob of bytes
        in a multipart envelope and pull in a parser to unwrap it.
        """
        contract = self.contract()
        duration = len(pcm) / SAMPLE_WIDTH / SAMPLE_RATE
        if duration < MIN_DURATION_SEC:
            return {"discarded": True, "duration": round(duration, 3)}

        spec = read_spec(dataset)
        declared = list(spec.get("languages") or [])
        src_lang = (src_lang or "").strip() or (declared[0] if declared else "en")
        if not SAFE_LANG.match(src_lang):
            raise DatasetError(400, f"언어 코드가 잘못됐다: {src_lang}")

        sessions = dataset / "data" / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)

        safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "", prefix or "") or "rec"
        base = f"{safe_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        taken = {r.get("id") for r in read_manifest(dataset)}
        item_id, n = base, 2
        while item_id in taken or (sessions / f"{item_id}.wav").exists():
            item_id, n = f"{base}_{n}", n + 1

        with wave.open(str(sessions / f"{item_id}.wav"), "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

        if not spec:
            # A hand-recorded dataset has no converter to write its spec, so the first
            # take bootstraps one.
            contract.write_spec(
                dataset,
                name=dataset.relative_to(self.root).as_posix(),
                split="record",
                languages=[src_lang],
                primary_metric="wer",
                translations=[],
                transcript=False,
                group_rule="id",
                bench_defaults={},
            )
        elif src_lang not in declared:
            # A dataset can hold more than one source language. Recording one the spec
            # has not declared widens the spec rather than contradicting it.
            spec["languages"] = declared + [src_lang]
            write_spec_raw(dataset, spec)

        row = contract.row(
            item_id=item_id,
            audio_rel=f"data/sessions/{item_id}.wav",
            duration=duration,
            src_lang=src_lang,
            transcript="",
            speaker=speaker or "",
        )
        # Not a contract field. It is the whole point of an overlap take, and it is
        # knowable only while recording, so it rides along.
        if speakers:
            row["speakers"] = speakers

        rows = read_manifest(dataset)
        rows.append(row)
        write_manifest(dataset, rows)
        return row
