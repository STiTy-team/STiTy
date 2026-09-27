"""Every run under bench/runs: what it is, which dataset it belongs to, and the
numbers the dashboard compares runs on."""

import json
from pathlib import Path

import numpy as np
from core.errors import DataError
from core.utils.json import read_json, read_jsonl
from core.utils.paths import get_project_root

from .. import config as bench_config
from ..config import get_runs_dir

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
    """Run directory names under the runs directory, most recently finished first."""

    def stamp(d: Path) -> float:
        summary = d / "summary.json"
        return summary.stat().st_mtime if summary.is_file() else 0.0

    dirs = [d for d in get_runs_dir().glob("*") if d.is_dir() and (d / "events.jsonl").is_file()]
    return [d.name for d in sorted(dirs, key=stamp, reverse=True)]


def read_summary(run_dir: Path) -> dict:
    try:
        return read_json(run_dir / "summary.json")
    except (OSError, json.JSONDecodeError):
        return {}


def pipeline_config(summary: dict) -> dict:
    """The resolved pipeline config the run actually used.

    `summary["config"]["stity"]` already holds every value the run actually used
    (including ones the source file left to a default) -- that beats reading
    `configs/pipelines/<name>.yml` back off disk, which only shows what was
    written and can drift from what the run historically saw.
    """
    return ((summary.get("config") or {}).get("stity")) or {}


def run_config(run_name: str, summary: dict) -> tuple[dict, str]:
    """The run's config and where it came from.

    A finished run recorded it in summary.json (`summary`). A run still going has no
    summary yet, so its pipeline and dataset configs are read back from `configs/`
    (`configs`) -- what the files say now, which is what the run started from unless
    someone edited them since.
    """
    if summary.get("config"):
        return summary["config"], "summary"
    pipeline, dataset_config = split_name(run_name)
    if not dataset_config:
        return {}, ""
    try:
        return bench_config.load(pipeline, dataset_config).raw, "configs"
    except Exception:  # noqa: BLE001 - a missing or broken config file just means no config
        return {}, ""


def run_catalog() -> list[dict]:
    """Every run with its dataset identity and config, so the client can offer it
    as a compare target without a second round trip once picked.

    `manifest_sha256` is a hash of the dataset's raw manifest.jsonl -- two runs
    share it only when their item ids are the same set, regardless of pipeline or
    which target language each was scored against.
    """
    out = []
    for name in list_runs():
        summary = read_summary(get_runs_dir() / name)
        config, source = run_config(name, summary)
        out.append(
            {
                "name": name,
                "dataset_sha": (summary.get("dataset") or {}).get("manifest_sha256") or "",
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


def _dataset_configs() -> list[str]:
    return sorted(
        (p.stem for p in (get_project_root() / "configs" / "datasets").glob("*.yml")),
        key=len,
        reverse=True,
    )


def split_name(run_name: str, known: list[str] | None = None) -> tuple[str, str]:
    """`<pipeline>-<dataset config>` back into its two halves.

    Both halves may contain dashes (`asr.qwen-seg-ko+mt.qwen3.5-4b`, `fleurs-ko-en`), so
    the dataset half is matched against the dataset configs that exist, longest first.
    """
    for stem in known if known is not None else _dataset_configs():
        if run_name.endswith("-" + stem) and len(run_name) > len(stem) + 1:
            return run_name[: -len(stem) - 1], stem
    return run_name, ""


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


def dataset_key(summary: dict, dataset_config: str) -> str:
    """Runs share a key only when they were scored on the same items and target."""
    if dataset_config:
        return dataset_config
    config = summary.get("config") or {}
    name = (config.get("dataset") or {}).get("name") or (summary.get("dataset") or {}).get("name")
    return f"{name or 'unknown'}-{config.get('target') or '?'}"


def _run_open(run_dir: Path) -> dict:
    """The run's first event -- all a run still going (no summary yet) can say about itself."""
    with open(run_dir / "events.jsonl", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") == "run_open":
                return event
    return {}


def run_overview(run_name: str, known: list[str]) -> dict:
    run_dir = get_runs_dir() / run_name
    summary = read_summary(run_dir)
    config, source = run_config(run_name, summary)
    dataset = summary.get("dataset") or {}
    pipeline, dataset_config = split_name(run_name, known)
    opened = {} if summary else _run_open(run_dir)
    ds_name = (
        (config.get("dataset") or {}).get("name")
        or dataset.get("name")
        or opened.get("dataset")
        or ""
    )
    counts = summary.get("counts") or {}
    if not summary and (run_dir / "items.jsonl").is_file():
        with open(run_dir / "items.jsonl", encoding="utf-8") as f:
            counts = {"items": sum(1 for line in f if line.strip()), "of": opened.get("n_items")}
    return {
        "name": run_name,
        "pipeline": pipeline,
        "dataset_key": dataset_key(summary, dataset_config),
        "dataset_name": ds_name,
        "corpus": ds_name.split("/")[0] if ds_name else "unknown",
        "target": config.get("target") or "",
        "limit": (config.get("dataset") or {}).get("limit"),
        "dataset_sha": dataset.get("manifest_sha256") or "",
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
    known = _dataset_configs()
    return {"runs": [run_overview(name, known) for name in list_runs()]}


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
