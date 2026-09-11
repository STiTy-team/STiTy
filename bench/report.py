"""Output files. The JSONL event stream is the source of truth; these are views.

Three files instead of one, because the old layout put the summary and every
per-utterance record in the same metric.json and reached 17-25 MB per run, which
is why results had to be untracked wholesale:

  results/<name>.json            summary only -- small enough to keep in git, so
                                 comparing two runs is a diff
  items/<name>-<stamp>.jsonl     one row per item, appended as the run goes, so a
                                 killed run is still scorable
  logs/<name>-<stamp>.jsonl      every event (written by events.py)
  logs/<name>-<stamp>.replay.json  the top-k items' events, projected from above

The summary is written at the end, not the start. The old meta.json was written
before the batch, so a resumed run overwrote the original run's arguments.
"""
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import events, metrics, paths


class ItemWriter:
    """Append-only per-item rows, flushed each time."""

    def __init__(self, path: Path, *, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done_ids: set[str] = set()
        if resume and self.path.exists():
            self.done_ids = {row.get("id") for row in events.read_stream(self.path)}
        self._file = open(self.path, "a" if resume else "w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def read_back(self) -> list[dict]:
        self._file.flush()
        return list(events.read_stream(self.path))

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=paths.PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def _package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "vllm", "transformers", "soundfile", "jiwer", "sacrebleu",
                 "PyYAML", "openai"):
        try:
            out[name.lower()] = md.version(name)
        except Exception:  # noqa: BLE001
            out[name.lower()] = None
    return out


def _gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def _artifact_path(path: Path) -> str:
    """Relative to bench/ when it lives there, absolute otherwise.

    Artifacts normally sit under bench/, but a caller may point them elsewhere
    (tests do), and relative_to raises rather than falling back.
    """
    try:
        return str(Path(path).relative_to(paths.BENCH_DIR))
    except ValueError:
        return str(path)


def _translator_mix(rows) -> dict:
    mix = Counter(seg.get("translator") or "unknown"
                  for row in rows for seg in row.get("segments") or [])
    return dict(sorted(mix.items()))


def select_top_k(rows, *, rank_by: str, order: str, top_k: int) -> list[dict]:
    """Worst (or best) k by the ranking metric, plus every failed item.

    Failures come first and are never crowded out by the ranking: an item that
    errored or produced no hypothesis at all is the one you opened the replay for.
    """
    key = metrics.primary_key(rank_by)
    failed = [r for r in rows if r.get("status") != "ok"
              or not (r.get("hypothesis") or "").strip()]
    failed_ids = {r.get("id") for r in failed}
    scored = [r for r in rows
              if r.get("id") not in failed_ids and r.get(key) is not None]
    scored.sort(key=lambda r: r[key], reverse=(order == "worst"))
    return failed + scored[:max(0, top_k - len(failed))]


def write_replay(events_path: Path, out_path: Path, chosen, *, cfg, stamp: str) -> Path:
    """Reads the event stream back and keeps only the chosen items' events."""
    buckets: dict[str, list] = {r.get("id"): [] for r in chosen}
    for event in events.read_stream(events_path):
        item_id = event.get("item")
        if item_id in buckets:
            buckets[item_id].append(event)

    data = []
    for rank, row in enumerate(chosen, start=1):
        item_id = row.get("id")
        data.append({
            "item": item_id,
            "rank": rank,
            "score": row.get(metrics.primary_key(cfg.logs.rank_by)),
            "status": row.get("status"),
            "reference": row.get("reference"),
            "hypothesis": row.get("hypothesis"),
            "events": buckets.get(item_id, []),
        })

    payload = {
        "name": cfg.name,
        "stamp": stamp,
        "fingerprint": cfg.fingerprint(),
        "timestamp_origin": "first_audio_chunk_of_item",
        "unit": "seconds",
        "rank_by": cfg.logs.rank_by,
        "order": cfg.logs.order,
        "top_k": cfg.logs.top_k,
        "data": data,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    return out_path


def write_all(*, cfg, dataset, rows, stamp, status, started: datetime,
              finished: datetime, events_path: Path, items_path: Path,
              usage: dict, translation: dict, transcription: dict) -> Path:
    aggregate, diagnostics = metrics.run_metrics(rows, cfg)

    errored = [r for r in rows if r.get("status") != "ok"]
    empty = [r for r in rows if r.get("status") == "ok"
             and not (r.get("hypothesis") or "").strip()]
    audio_sec = sum(float(r.get("audio_sec") or 0) for r in rows)
    wall_sec = (finished - started).total_seconds()

    if errored or diagnostics["uncomputable"]:
        status = "degraded"

    warnings = []
    if empty:
        warnings.append(
            f"{len(empty)} item(s) produced an empty hypothesis: "
            + ", ".join(r.get("id") for r in empty[:5]))
    if errored:
        warnings.append(f"{len(errored)} item(s) failed: "
                        + ", ".join(r.get("id") for r in errored[:5]))
    mix = _translator_mix(rows)
    if translation.get("backend") == "gpt" and mix:
        non_gpt = sum(n for k, n in mix.items() if k != "gpt")
        if non_gpt:
            warnings.append(
                f"translation 'gpt' but {non_gpt} of {sum(mix.values())} commits did "
                f"not use it: the server skips the GPT path for the first commit of "
                f"every stream (_committed_utterance_count > 0)")
    late = sum(int(r.get("n_foreign_finals") or 0) for r in rows)
    if late:
        warnings.append(f"{late} final(s) arrived attributed to another item")

    chosen = select_top_k(rows, rank_by=cfg.logs.rank_by, order=cfg.logs.order,
                          top_k=cfg.logs.top_k)
    replay_path = events_path.with_suffix(".replay.json")
    write_replay(events_path, replay_path, chosen, cfg=cfg, stamp=stamp)

    payload = {
        "name": cfg.name,
        "status": status,
        "fingerprint": cfg.fingerprint(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "metrics": aggregate,
        "counts": {
            "n_items": len(rows),
            "n_sessions": len({r.get("group") for r in rows if r.get("group")}),
            "n_errored": len(errored),
            "n_empty_hypothesis": len(empty),
            "n_late_finals": late,
            "audio_sec": round(audio_sec, 2),
            "wall_sec": round(wall_sec, 2),
            "realtime_factor": round(audio_sec / wall_sec, 3) if wall_sec else None,
        },
        "translator_mix": mix,
        "config": {"raw": cfg.raw, "overrides": cfg.overrides, "resolved": cfg.resolved()},
        "dataset": dataset.provenance(),
        "components": {"transcription": transcription, "translation": translation},
        "env": {**_git_state(), **_package_versions(), "gpu": _gpu_name()},
        "cost": usage,
        "diagnostics": {**diagnostics, "warnings": warnings},
        "artifacts": {
            "events": _artifact_path(events_path),
            "items": _artifact_path(items_path),
            "replay": _artifact_path(replay_path),
        },
    }

    out_path = paths.RESULTS_DIR / f"{cfg.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print_summary(payload)
    return out_path


def print_summary(payload: dict) -> None:
    m = payload["metrics"]
    c = payload["counts"]
    print()
    print(f"{payload['name']}  [{payload['status']}]  fingerprint={payload['fingerprint']}")
    print(f"  items {c['n_items']}  sessions {c['n_sessions']}  "
          f"errored {c['n_errored']}  empty {c['n_empty_hypothesis']}")
    for key in ("wer", "wer_scored_only", "cer", "bleu", "bleu_all", "laal_ms",
                "avg_fsl_sec", "lang_detect_accuracy", "route_accuracy", "n_misrouted"):
        if key in m:
            value = m[key]
            shown = "-" if value is None else (f"{value:.4f}" if isinstance(value, float)
                                               else value)
            print(f"  {key:22} {shown}")
    stats = m.get("commit_stats")
    if stats:
        counts = {k: v for k, v in stats["counts"].items() if v}
        print(f"  commit                 {counts} (finish {stats['finish_ratio']:.1%})")
    if payload["translator_mix"]:
        print(f"  translator_mix         {payload['translator_mix']}")
    cost_info = payload.get("cost") or {}
    if cost_info.get("calls"):
        print(f"  api calls              {cost_info['calls']} "
              f"(in {cost_info['input_tokens']}, out {cost_info['output_tokens']})")
    for warning in payload["diagnostics"]["warnings"]:
        print(f"  ! {warning}")
    for metric, reason in payload["diagnostics"]["uncomputable"].items():
        print(f"  ! {metric} not computed: {reason}")
    print(f"  -> {paths.RESULTS_DIR / (payload['name'] + '.json')}")
