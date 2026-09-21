import json
import subprocess
from datetime import datetime
from pathlib import Path

from core.utils import logging


class ItemWriter:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def read_back(self) -> list[dict]:
        self._file.flush()
        return list(logging.read_stream(self.path))

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def environment() -> dict:
    repo = Path(__file__).resolve().parent.parent

    def git(*args) -> str:
        try:
            return subprocess.run(("git", *args), cwd=repo, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001 - a run outside a checkout still runs
            return ""

    import importlib.metadata as md

    versions = {}
    for name in ("torch", "vllm", "transformers", "sacrebleu", "unbabel-comet"):
        try:
            versions[name] = md.version(name)
        except Exception:  # noqa: BLE001 - not installed is an answer
            versions[name] = None

    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass

    return {"commit": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "gpu": gpu, **versions}


def write_all(*, cfg, dataset, score, rows, stamp, status, started: datetime,
              finished: datetime, run_dir: Path, components: dict, pacing: dict,
              failure: str | None = None) -> Path:
    errored = [r for r in rows if r.get("status") != "ok"]
    empty = [r for r in rows if r.get("status") == "ok"
             and not (r.get("hypothesis") or "").strip()]
    audio_sec = sum(float(r.get("audio_sec") or 0) for r in rows)
    wall_sec = (finished - started).total_seconds()

    payload = {
        "name": cfg.name,
        "stamp": stamp,
        "status": status if status != "ok" else ("degraded" if errored else "ok"),
        "failure": failure,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "metrics": score.aggregate,
        "unavailable": score.unavailable,
        "counts": {
            "items": len(rows),
            "sessions": len({r.get("group") for r in rows if r.get("group")}),
            "errored": len(errored),
            "empty_hypothesis": len(empty),
            "audio_sec": round(audio_sec, 2),
            "wall_sec": round(wall_sec, 2),
            "realtime_factor": round(audio_sec / wall_sec, 3) if wall_sec else None,
        },
        "failed_items": [r.get("id") for r in errored + empty],
        "misrouted_items": score.misrouted_items,
        "config": cfg.raw,
        "pacing": pacing,
        "dataset": dataset.provenance(),
        "components": components,
        "environment": environment(),
    }

    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print_summary(payload, out_path)
    return out_path


def write_translation_only(*, cfg, score, rows, source: dict, started: datetime,
                           finished: datetime, run_dir: Path,
                           component: dict, failure: str | None = None) -> Path:
    """Write a summary for a stored-ASR translation replay.

    ASR latency is intentionally absent: its clocks came from the source run and
    were not measured during this execution.
    """
    errored = [row for row in rows if row.get("translation_status") not in
               ("ok", "skipped_source_error", "skipped_empty_source")]
    skipped_source_error = [row for row in rows
                            if row.get("translation_status") == "skipped_source_error"]
    skipped_empty_source = [row for row in rows
                            if row.get("translation_status") == "skipped_empty_source"]
    empty = [row for row in rows if row.get("translation_calls", 0)
             and not (row.get("hypothesis_translation") or "").strip()]
    failed_ids = list(dict.fromkeys(
        str(row.get("id") or "")
        for row in errored + skipped_source_error + skipped_empty_source + empty))
    wall_sec = (finished - started).total_seconds()
    status = "failed" if failure else ("degraded" if failed_ids else "ok")
    payload = {
        "name": cfg.name,
        "mode": "translation_only",
        "stamp": started.strftime("%Y%m%dT%H%M%S"),
        "status": status,
        "failure": failure,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "metrics": score.aggregate,
        "unavailable": score.unavailable,
        "counts": {
            "items": len(rows),
            "sessions": len({row.get("group") for row in rows if row.get("group")}),
            "segments": sum(len(row.get("segments") or []) for row in rows),
            "translation_errors": sum(len(row.get("translation_errors") or [])
                                      for row in rows),
            "empty_translation": len(empty),
            "source_error_skips": len(skipped_source_error),
            "empty_source_skips": len(skipped_empty_source),
            "wall_sec": round(wall_sec, 2),
        },
        "failed_items": failed_ids,
        "misrouted_items": score.misrouted_items,
        "config": cfg.raw,
        "components": {"translation": component},
        "source_run": source,
        "measurement_scope": {
            "asr": "frozen from source run; WER/FSL/LAAL not rescored",
            "segmentation": "frozen committed segments from source run",
            "translation": "executed in this run",
        },
        "environment": environment(),
    }
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print_summary(payload, out_path)
    return out_path


def _show(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return "  ".join(f"{k}={_show(v)}" for k, v in value.items())
    return str(value)


def print_summary(payload: dict, out_path: Path) -> None:
    counts = payload["counts"]
    print()
    print(f"{payload['name']}  [{payload['status']}]")
    print("  " + "  ".join(f"{k} {v}" for k, v in counts.items() if v is not None))
    for key, value in sorted(payload["metrics"].items()):
        has_nested_entries = (isinstance(value, dict)
                              and any(isinstance(v, dict) for v in value.values()))
        if has_nested_entries:
            print(f"  {key}")
            for name, inner in value.items():
                print(f"    {name:20} {_show(inner)}")
        else:
            print(f"  {key:22} {_show(value)}")
    for metric, reason in payload["unavailable"].items():
        print(f"  - {metric} unavailable: {reason}")
    if payload.get("failure"):
        print(f"  ! run failed: {payload['failure']}")
    if payload["failed_items"]:
        print(f"  ! failed: {', '.join(payload['failed_items'][:5])}")
    print(f"  -> {out_path}")
