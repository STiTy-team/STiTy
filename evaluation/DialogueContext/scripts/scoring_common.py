"""Shared helpers for the DialogueContext scorers and the aggregator.

The run directory holds `translations.jsonl` (one row per job, see DESIGN.md "결과 행").
Scorers read it, never modify it, and write their own `scores_*.jsonl` keyed by `job_id`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_INSTANCES = EXPERIMENT_DIR / "data" / "instances.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_env() -> None:
    """Read the repo-root `.env` into the process environment (the shell wins)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def env_value(name: str) -> str:
    """A key from the process environment, else straight from `.env`, stripped."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from dotenv import dotenv_values
    except ImportError:
        return ""
    return (dotenv_values(REPO_ROOT / ".env").get(name) or "").strip()


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A crash mid-append leaves one torn last line; everything before it is good.
                print(f"[warn] {path}:{number} is not valid JSON, skipped", file=sys.stderr)
    return rows


def append_jsonl(path: str | Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: str | Path, rows) -> None:
    """Replace the file atomically so a reader never sees half of it."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(*parts) -> str:
    joined = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def resolve_instances_path(run_dir: Path, explicit: str | None) -> Path:
    """`--instances` wins, then a snapshot inside the run dir, then data/instances.jsonl."""
    if explicit:
        return Path(explicit)
    for candidate in (run_dir / "instances.jsonl", run_dir / "data" / "instances.jsonl"):
        if candidate.exists():
            return candidate
    return DEFAULT_INSTANCES


def load_instances(path: Path) -> dict[str, dict]:
    return {row["instance_id"]: row for row in read_jsonl(path)}


def scorable_rows(rows: list[dict], instances: dict[str, dict] | None = None) -> list[dict]:
    """Rows that carry a translation of an evaluation target turn.

    Warm-up rows, failed jobs (no hypothesis) and S1 rows for non-target turns
    (no instance / no reference) are left out.
    """
    out = []
    for row in rows:
        if row.get("phase") == "warmup":
            continue
        if row.get("error") and row.get("hypothesis") is None:
            continue
        if row.get("hypothesis") is None or not row.get("reference"):
            continue
        if instances is not None and row.get("instance_id") not in instances:
            continue
        out.append(row)
    return out


def latest_by_job(rows: list[dict]) -> dict[str, dict]:
    """A restarted runner may append the same job twice; the last line wins."""
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("job_id") is not None:
            out[row["job_id"]] = row
    return out
