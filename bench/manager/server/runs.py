"""Every run under bench/runs: what it is, which dataset it belongs to, and the
numbers the dashboard compares runs on."""

import json
from pathlib import Path

import numpy as np
from core import config as core_config
from core.errors import ConfigError, DataError
from core.utils.json import read_jsonl

from ... import registry
from ...config import get_runs_dir
from ...registry import read_summary

# Per-item scores the dashboard draws as distributions. `comet` has no per-item
# value in items.jsonl -- it comes per sentence from comet_scores.jsonl.
DISTRIBUTION_KEYS = (
    "wer",
    "cer",
    "sentence_bleu",
    "comet",
    "avg_fsl_sec",
    "token_emission_ms",
    "token_emission_ca_ms",
    "laal_ms",
    "laal_ca_ms",
    "yaal_ms",
    "yaal_ca_ms",
    "longyaal_ms",
    "longyaal_ca_ms",
    "n_segments",
)
# A run of thousands of items would put every value on the page; this many
# points, evenly spaced through the sorted values, keep the shape.
MAX_POINTS = 240
COMET_SCORES = "comet_scores.jsonl"


def list_runs() -> list[str]:
    """Run names (`<dataset ref>/<pipeline ref>`), most recently finished first."""

    def stamp(name: str) -> float:
        summary = get_runs_dir() / name / "summary.json"
        return summary.stat().st_mtime if summary.is_file() else 0.0

    return sorted(registry.run_names(), key=stamp, reverse=True)


def run_config(run_dir: Path, summary: dict) -> tuple[dict, str]:
    """The config the run recorded and where: `summary` once it finished, `events`
    (its `run_open` line) while it is still going. Never read back from `configs/`,
    which only says what the files hold now."""
    seen = registry.recorded(run_dir, summary)
    return seen["config"], seen["source"] if seen["config"] else ""


def _current_meta(kind: str, ref: str) -> dict | None:
    name, _, version = ref.partition("@v")
    try:
        _, meta = core_config.read_named_with_meta(name, kind)
    except ConfigError:
        return None
    return meta.model_dump() if meta.version == int(version or 1) else None


def run_meta(run_dir: Path, summary: dict) -> dict:
    """Per kind: the ref, version and hash the run recorded, with the description and
    tags the config file has now when it is still at that version -- tags added later
    (`paper`, say) show on old runs too -- else the ones the run recorded."""
    identity = registry.recorded(run_dir, summary)["identity"]
    refs = dict(zip(registry.KINDS, registry.split(registry.run_name(run_dir))))
    out = {}
    for kind in registry.KINDS:
        seen = identity.get(kind) or {"ref": refs[kind]}
        now = _current_meta(kind, seen["ref"]) or {}
        out[kind] = {
            "ref": seen["ref"],
            "version": seen.get("version") or now.get("version") or 1,
            "hash": seen.get("hash") or "",
            "description": now.get("description", seen.get("description") or ""),
            "tags": now.get("tags", seen.get("tags") or []),
        }
    return out


def data_key(config: dict) -> str:
    """What a run was fed: the dataset config minus the pipeline.

    Two runs share it only when every item reached the pipeline as the same audio
    with the same target -- `fleurs_ko-en` and `fleurs_ko-en_cafe` read the same
    manifest but play different sound, so they must not be compared item by item.
    """
    fed = {k: v for k, v in config.items() if k not in ("name", "stity")}
    return json.dumps(fed, sort_keys=True) if fed.get("dataset") else ""


def item_ids(run_dir: Path) -> list[str]:
    """The ids of the items this run has a row for (a running run: so far)."""
    path = run_dir / "items.jsonl"
    if not path.is_file():
        return []
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            head = line[:200]
            start = head.find('"id": "')
            if start != -1:
                start += len('"id": "')
                ids.append(head[start : head.index('"', start)])
    return ids


def run_catalog() -> list[dict]:
    """Every run with its dataset identity and config, so the client can offer it
    as a compare target without a second round trip once picked.

    A compare target must have been fed the same data (`data_key`, and the same
    `manifest_sha256` when both runs recorded one), run a different pipeline, and
    have a row for the item on screen (`item_ids`).
    """
    out = []
    for name in list_runs():
        run_dir = get_runs_dir() / name
        summary = read_summary(run_dir)
        config, source = run_config(run_dir, summary)
        dataset, pipeline = registry.split(name)
        out.append(
            {
                "name": name,
                "dataset": dataset,
                "pipeline": pipeline,
                "meta": run_meta(run_dir, summary),
                "dataset_sha": registry.recorded(run_dir, summary)["manifest_sha256"],
                "data_key": data_key(config),
                "item_ids": item_ids(run_dir),
                "pipeline_config": config.get("stity") or {},
                "config_source": source,
                "status": summary.get("status") or ("running" if not summary else ""),
                "summary_metrics": summary.get("metrics") or {},
            }
        )
    return out


def resolve_run(requested: str | None) -> str:
    """The requested run if it exists, else the most recently finished one."""
    runs = list_runs()
    if not runs:
        raise DataError(f"no runs yet in {get_runs_dir()}")
    if requested and requested in runs:
        return requested
    return runs[0]


def _models(summary: dict) -> dict:
    """The model behind each stage, for the leaderboard's second line."""
    components = summary.get("components") or {}
    out = {}
    for stage in ("transcription", "correction", "translation"):
        part = components.get(stage) or {}
        model = part.get("model_path") or part.get("model") or part.get("backend")
        if model:
            out[stage] = str(model)
    return out


def run_overview(run_name: str) -> dict:
    run_dir = get_runs_dir() / run_name
    summary = read_summary(run_dir)
    config, source = run_config(run_dir, summary)
    dataset_ref, pipeline_ref = registry.split(run_name)
    opened = {} if summary else registry.run_open(run_dir)
    dataset_config = config.get("dataset") or {}
    counts = summary.get("counts") or {}
    if not summary and (run_dir / "items.jsonl").is_file():
        with open(run_dir / "items.jsonl", encoding="utf-8") as f:
            counts = {"items": sum(1 for line in f if line.strip()), "of": opened.get("n_items")}
    return {
        "name": run_name,
        "pipeline": pipeline_ref,
        "dataset_key": dataset_ref,
        "dataset_name": dataset_config.get("name") or opened.get("dataset") or "",
        "corpus": dataset_ref.split("_")[0],
        "target": config.get("target") or "",
        "limit": dataset_config.get("limit"),
        "pick": dataset_config.get("pick") or "first",
        "meta": run_meta(run_dir, summary),
        "dataset_sha": registry.recorded(run_dir, summary)["manifest_sha256"],
        "stamp": summary.get("stamp") or "",
        "finished_at": summary.get("finished_at") or "",
        "status": summary.get("status") or ("running" if not summary else ""),
        "counts": counts,
        "models": _models(summary),
        "pipeline_config": config.get("stity") or {},
        "config_source": source,
        "summary_metrics": summary.get("metrics") or {},
        "unavailable": summary.get("unavailable") or {},
    }


def overview() -> dict:
    """Everything the dashboard needs up front: every run, tagged with its dataset."""
    return {"runs": [run_overview(name) for name in list_runs()]}


def _stats(values: list[float]) -> dict:
    a = np.sort(np.asarray(values, dtype=float))
    q1, median, q3 = np.percentile(a, [25, 50, 75])
    if a.size > MAX_POINTS:
        points = a[np.linspace(0, a.size - 1, MAX_POINTS).round().astype(int)]
    else:
        points = a
    return {
        "n": int(a.size),
        "min": float(a[0]),
        "q1": float(q1),
        "median": float(median),
        "mean": float(a.mean()),
        "q3": float(q3),
        "max": float(a[-1]),
        "points": [round(float(v), 4) for v in points],
    }


_distribution_cache: dict[str, tuple[tuple, dict]] = {}


def distributions(run_name: str) -> dict:
    """Per-item score distributions of one run, keyed like DISTRIBUTION_KEYS.

    Only items that finished (`status == ok`) count, and a key is left out when no
    item has it -- a run without transcripts has no WER to draw.
    """
    run_dir = get_runs_dir() / run_name
    items, scores = run_dir / "items.jsonl", run_dir / COMET_SCORES
    stamp = tuple(p.stat().st_mtime if p.is_file() else 0.0 for p in (items, scores))
    cached = _distribution_cache.get(run_name)
    if cached and cached[0] == stamp:
        return cached[1]

    values: dict[str, list[float]] = {key: [] for key in DISTRIBUTION_KEYS}
    if items.is_file():
        for row in read_jsonl(items):
            if row.get("status") != "ok":
                continue
            for key in DISTRIBUTION_KEYS:
                v = row.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                    values[key].append(float(v))
    if scores.is_file():
        values["comet"] = [
            float(r["comet"]) for r in read_jsonl(scores) if r.get("comet") is not None
        ]
    out = {key: _stats(v) for key, v in values.items() if v}
    _distribution_cache[run_name] = (stamp, out)
    return out
