"""One run, one item at a time: the data behind the session replay page."""

import hashlib
import io
import json
import math
import urllib.parse
from functools import lru_cache
from pathlib import Path

import numpy as np
from core.errors import DataError, STiTyError
from core.utils import audio
from core.utils.json import read_jsonl

from ... import augment
from ...config import DatasetConfig, get_runs_dir
from ...settings import BenchSettings
from ...registry import run_name as name_of
from .runs import list_runs, read_summary, run_catalog, run_config, run_meta

ITEM_METRIC_KEYS = (
    "wer",
    "cer",
    "sentence_bleu",
    "avg_fsl_sec",
    "laal_ms",
    "yaal_ms",
    "longyaal_ms",
    "token_emission_ms",
    "n_segments",
)
# Worst first: the first key an item has decides its place. Runs without transcripts
# rank by translation, runs without any reference by latency.
RANKING = (("wer", True), ("sentence_bleu", False), ("avg_fsl_sec", True))

AUDIO_API = "/api/replay/audio"
DEFAULT_TOP_K = 10


def choose_items(rows: dict[str, dict], *, top_k: int) -> list[str]:
    """Failed items and items with no transcription output first, then the worst of the rest, up to top_k.

    A failure is never crowded out by the ranking -- it is the reason the replay
    got opened. See bench/README.md's replay contract.
    """
    failed = [
        r
        for r in rows.values()
        if r.get("status") != "ok" or not (r.get("transcription_output") or "").strip()
    ]
    failed_ids = {r["id"] for r in failed}
    rest = [r for r in rows.values() if r["id"] not in failed_ids]
    return [r["id"] for r in failed + sorted(rest, key=_badness)][: max(top_k, len(failed))]


def _badness(row: dict) -> tuple:
    for rank, (key, higher_is_worse) in enumerate(RANKING):
        if row.get(key) is not None:
            return rank, -row[key] if higher_is_worse else row[key]
    return len(RANKING), 0


def dataset_root(summary: dict, run_dir: Path | None = None) -> Path | None:
    """The run's dataset directory: where the summary says, else under the data root.

    A run with no summary yet still names its audio in items.jsonl, and every audio
    file sits inside its dataset -- so the nearest folder above one that holds a
    manifest.jsonl is the dataset.
    """
    declared = str((summary.get("dataset") or {}).get("root") or "")
    candidates = []
    if declared:
        candidates.append(Path(declared))
        try:
            candidates.append(BenchSettings.load().stity_data_root / Path(declared).name)
        except STiTyError:
            pass
    found = next((c for c in candidates if (c / "manifest.jsonl").is_file()), None)
    if found is not None or run_dir is None or not (run_dir / "items.jsonl").is_file():
        return found
    row = next((r for r in read_jsonl(run_dir / "items.jsonl") if r.get("audio")), None)
    if row is None:
        return None
    return next((p for p in Path(row["audio"]).parents if (p / "manifest.jsonl").is_file()), None)


def audio_index(summary: dict, run_dir: Path | None = None) -> dict[str, dict]:
    ds_root = dataset_root(summary, run_dir)
    if ds_root is not None:
        manifest = ds_root / "manifest.jsonl"
        index = {}
        for raw in read_jsonl(manifest):
            item_id = raw.get("id")
            if not item_id or not raw.get("audio"):
                continue
            path = ds_root / str(raw["audio"])
            index[str(item_id)] = {
                "path": path,
                "offset": raw.get("offset"),
                "duration": raw.get("duration"),
            }
            if raw.get("offset") is not None and raw.get("group"):
                index.setdefault(
                    str(raw["group"]), {"path": path, "offset": None, "duration": None}
                )
        if index:
            return index
    return {}


def run_clips(run_dir: Path) -> dict[str, dict]:
    """Every item's audio for one run: the dataset manifest, then the run's own rows.

    The manifest is found through `summary.json`, which only a finished run has.
    Each `items.jsonl` row records the clip the bench itself played, so a run that is
    still going or died before its summary still has sound.
    """
    index = audio_index(read_summary(run_dir), run_dir)
    if (run_dir / "items.jsonl").is_file():
        for row in read_jsonl(run_dir / "items.jsonl"):
            if row.get("id") and row.get("audio") and str(row["id"]) not in index:
                index[str(row["id"])] = {
                    "path": Path(row["audio"]),
                    "offset": row.get("offset"),
                    "duration": row.get("duration_sec"),
                }
            if row.get("augment") and str(row.get("id")) in index:
                index[str(row["id"])]["augment"] = row["augment"]
    return index


@lru_cache(maxsize=8)
def run_augmenter(run_dir: Path) -> tuple[augment.Augmenter, str]:
    config, _ = run_config(run_dir, read_summary(run_dir))
    spec = config.get("augment")
    if not spec:
        return augment.Augmenter([]), ""
    dataset = DatasetConfig.parse(config["dataset"]).name
    return augment.build(
        augment.AugmentConfig.parse(spec), data_root=BenchSettings.load().stity_data_root
    ), dataset


def clip_samples(run_dir: Path, item_id: str, entry: dict) -> np.ndarray:
    """The clip as the bench fed it: the file, then the run's augmentation rebuilt.

    The augmentation is deterministic per item, so rebuilding it gives the same audio
    the pipeline heard. The choices recorded in the row prove it -- if the noise or
    room files changed since the run, they no longer match and the clip is refused
    rather than played clean or different.
    """
    samples = audio.load_window(
        entry["path"], offset=entry.get("offset"), duration=entry.get("duration")
    )
    recorded = entry.get("augment")
    if not recorded:
        return samples
    augmenter, dataset = run_augmenter(Path(run_dir))
    samples, choices = augmenter.apply(samples, key=augment.item_key(dataset, item_id))
    if choices != recorded:
        raise DataError(
            f"rebuilt augmentation for {item_id!r} does not match the run "
            f"(recorded {recorded}, rebuilt {choices}); the augment files changed since"
        )
    return samples


def _dbfs(x: float) -> float:
    return round(20 * math.log10(max(x, 1e-6)), 1)


def item_data(run_dir: Path, item_id: str) -> dict | None:
    """What the dataset says about one item: its turns, their levels and word timings,
    the audio file behind it, and the manifest rows it came from.

    A conversation is one item made of several manifest rows (one per turn, sharing a
    `group`). Without a manifest -- a run still going, or one that died before its
    summary -- the turns come from the run's own `reference_segmentation` instead.
    """
    run_dir = Path(run_dir)
    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in read_jsonl(run_dir / "items.jsonl")}
    row = rows.get(item_id)
    clip = run_clips(run_dir).get(item_id)
    if row is None and clip is None:
        return None
    row = row or {}
    summary = read_summary(run_dir)
    ds_root = dataset_root(summary, run_dir)

    raw_rows = []
    if ds_root is not None:
        with open(ds_root / "manifest.jsonl", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("id") == item_id or (
                    raw.get("offset") is not None and raw.get("group") == item_id
                ):
                    raw_rows.append({"line": line_no, "row": raw})

    clip_start = float((clip or {}).get("offset") or 0.0)
    turns = []
    if raw_rows:
        for r in (x["row"] for x in raw_rows):
            ref = r.get("reference") or {}
            start = float(r.get("offset") or 0.0) - clip_start
            turns.append(
                {
                    "id": r.get("id"),
                    "start": start,
                    "end": start + float(r.get("duration") or 0.0),
                    "src_lang": r.get("src_lang") or "",
                    "speaker": r.get("speaker") or "",
                    "partial": bool(r.get("partial")),
                    "transcript": ref.get("transcript") or "",
                    "translations": ref.get("translations") or {},
                }
            )
    else:
        segments = row.get("reference_segmentation") or [
            {
                "offset": 0.0,
                "duration": row.get("duration_sec"),
                "src_lang": row.get("src_lang"),
                "partial": row.get("partial"),
                "transcript": row.get("reference"),
                "translations": row.get("reference_translations"),
            }
        ]
        for i, seg in enumerate(segments):
            start = float(seg.get("offset") or 0.0)
            turns.append(
                {
                    "id": f"{item_id}#{i}",
                    "start": start,
                    "end": start + float(seg.get("duration") or 0.0),
                    "src_lang": seg.get("src_lang") or "",
                    "speaker": "",
                    "partial": bool(seg.get("partial")),
                    "transcript": seg.get("transcript") or "",
                    "translations": seg.get("translations") or {},
                }
            )

    audio_info = None
    if clip is not None:
        audio_info = {
            "path": str(clip["path"]),
            "offset": clip.get("offset"),
            "duration": clip.get("duration"),
            "augment": clip.get("augment"),
        }
        try:
            import soundfile as sf

            info = sf.info(str(clip["path"]))
            audio_info.update(
                format=f"{info.format} {info.subtype}",
                sample_rate=info.samplerate,
                channels=info.channels,
                file_sec=round(info.duration, 3),
            )
            samples = clip_samples(run_dir, item_id, clip)
            for turn in turns:
                window = samples[
                    int(turn["start"] * audio.SAMPLING_RATE) : int(
                        turn["end"] * audio.SAMPLING_RATE
                    )
                ]
                if window.size:
                    turn["peak_dbfs"] = _dbfs(float(np.abs(window).max()))
                    turn["rms_dbfs"] = _dbfs(float(np.sqrt(np.mean(window**2))))
        except (STiTyError, OSError, RuntimeError) as e:
            audio_info["error"] = str(e)

    spec = ""
    if ds_root is not None and (ds_root / "dataset.yml").is_file():
        spec = (ds_root / "dataset.yml").read_text(encoding="utf-8")
    dataset = summary.get("dataset") or {}
    name = dataset.get("name") or (run_config(run_dir, summary)[0].get("dataset") or {}).get("name")
    sha = dataset.get("manifest_sha256")
    if not sha and ds_root is not None:
        # A finished run records the hash; one still going has not yet, so take it
        # the same way bench does: sha256 of the raw manifest file.
        sha = hashlib.sha256((ds_root / "manifest.jsonl").read_bytes()).hexdigest()
    return {
        "id": item_id,
        "dataset": {
            "name": name or "",
            "root": str(ds_root) if ds_root else "",
            "manifest_sha256": sha or "",
            "spec": spec,
        },
        "audio": audio_info,
        "turns": turns,
        "words": row.get("reference_alignment") or [],
        "manifest_rows": raw_rows,
    }


def wav_bytes(samples: np.ndarray) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, samples, audio.SAMPLING_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _at_commit(event: dict) -> dict:
    if event.get("type") != "transcribed":
        return event
    return {
        **event,
        "t": event.get("committed_elapsed_sec", event.get("t")),
        "audio": event.get("decision_audio_sec", event.get("audio")),
    }


def _finalize_item(
    item_id: str,
    events: list[dict],
    duration: float,
    *,
    rows: dict[str, dict],
    clips: dict[str, dict],
    run_name: str,
) -> dict:
    opened = next((e for e in events if e.get("type") == "item_open"), {})
    item = {
        "id": item_id,
        "events": events,
        "duration": duration,
        "src_lang": opened.get("src_lang") or "",
        "target_lang": opened.get("target_lang") or "",
    }
    row = rows.get(item_id) or {}
    wer = row.get("wer")
    if wer is not None:
        item["wer"] = wer
    metrics = {k: row[k] for k in ITEM_METRIC_KEYS if row.get(k) is not None}
    if metrics:
        item["metrics"] = metrics
    item["reference"] = row.get("reference") or ""
    item["reference_translations"] = row.get("reference_translations") or {}
    played_past_the_clip = [e.get("audio") or 0.0 for e in events]
    item["duration"] = max([item["duration"]] + played_past_the_clip)
    if item_id in clips:
        item["audio_url"] = f"{AUDIO_API}?{urllib.parse.urlencode({'run': run_name, 'item': item_id})}"
    return item


def item_payload(run_dir: Path, item_id: str) -> dict | None:
    """One item's data from a run, regardless of whether it made that run's own cut.

    Backs the "compare with" picker: the item on screen may not be among the
    other run's worst-`top_k`, so it has to be fetched on demand instead of
    riding along in that run's own page load.
    """
    run_dir = Path(run_dir)
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return None

    events: list[dict] = []
    duration = 0.0
    found = False
    for event in read_jsonl(events_path):
        if event.get("item") != item_id:
            continue
        found = True
        events.append(_at_commit(event))
        if event.get("type") == "item_open":
            duration = float(event.get("audio_sec") or 0.0)
    if not found:
        return None

    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in read_jsonl(run_dir / "items.jsonl")}
    clips = run_clips(run_dir)
    return _finalize_item(
        item_id, events, duration, rows=rows, clips=clips, run_name=name_of(run_dir)
    )


def payload(run_dir: Path, *, top_k: int = DEFAULT_TOP_K) -> dict:
    """One run's replay data: the worst-`top_k` items by `RANKING`, plus every failure.

    Only the first item ships with its events; the page fetches any other one from
    `/item/` when it is picked. A run of long talks would otherwise put every
    chosen talk's event stream into one page.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        siblings = list_runs()
        raise DataError(
            f"{run_dir} is not a directory. Runs in {get_runs_dir()}: "
            + (", ".join(siblings) if siblings else "none yet")
        )
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise DataError(f"{events_path} is missing -- replay reads a run's event stream")

    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in read_jsonl(run_dir / "items.jsonl")}
    summary = read_summary(run_dir)

    worst = choose_items(rows, top_k=top_k)
    first = (
        worst[0]
        if worst
        else next((e["item"] for e in read_jsonl(events_path) if e.get("item")), None)
    )
    item = item_payload(run_dir, first) if first else None
    if item is None:
        raise DataError(f"{events_path} has no item events")

    config, config_source = run_config(run_dir, summary)
    catalog = run_catalog()
    return {
        "run": {
            "name": name_of(run_dir),
            "stamp": summary.get("stamp") or "",
            "status": summary.get("status") or ("running" if not summary else ""),
            "dir": name_of(run_dir),
            "meta": run_meta(run_dir, summary),
            "dataset_sha": next(
                (r["dataset_sha"] for r in catalog if r["name"] == name_of(run_dir)), ""
            ),
            "n_total": len(rows),
            "top_k": top_k,
            "pipeline_config": config.get("stity") or {},
            "config_source": config_source,
            "summary_metrics": summary.get("metrics") or {},
        },
        "items": [item],
        "worst": [{"id": i, "wer": rows[i].get("wer")} for i in worst],
        "all_items": [{"id": r["id"], "wer": r.get("wer")} for r in rows.values()],
        "runs": catalog,
    }

