"""The runs on disk, as `bench/runs/<dataset ref>/<pipeline ref>/`, and the rule that
keeps a config name meaning one thing: a ref that has runs may not change its config
or its data without a new `meta.version`."""

import json
from pathlib import Path

from core.errors import ConfigError
from core.integrations.s3 import Bucket
from core.utils import logging
from core.utils.json import read_json

from . import store
from .config import get_runs_dir
from .settings import BenchSettings

log = logging.getLogger(__name__)

KINDS = ("dataset", "pipeline")


def run_names() -> list[str]:
    root = get_runs_dir()
    return sorted(
        d.relative_to(root).as_posix()
        for d in root.glob("*/*")
        if d.is_dir() and (d / "events.jsonl").is_file()
    )


def run_name(run_dir: Path) -> str:
    return run_dir.relative_to(get_runs_dir()).as_posix()


def split(name: str) -> tuple[str, str]:
    dataset, _, pipeline = name.partition("/")
    return dataset, pipeline


def read_summary(run_dir: Path) -> dict:
    try:
        return read_json(run_dir / "summary.json")
    except (OSError, json.JSONDecodeError):
        return {}


def run_open(run_dir: Path) -> dict:
    """The run's first event -- all a run still going (no summary yet) can say about itself."""
    try:
        with open(run_dir / "events.jsonl", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if event.get("type") == "run_open":
                    return event
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def recorded(run_dir: Path, summary: dict | None = None) -> dict:
    """What the run wrote down about itself: `config`, `identity` and `manifest_sha256`,
    from its summary when it has one, else from its `run_open` event."""
    summary = read_summary(run_dir) if summary is None else summary
    source = summary if summary.get("config") else run_open(run_dir)
    return {
        "config": source.get("config") or {},
        "identity": source.get("identity") or {},
        "manifest_sha256": (
            (summary.get("dataset") or {}).get("manifest_sha256")
            or source.get("manifest_sha256")
            or ""
        ),
        "source": "summary" if source is summary else "events",
    }


def check(
    identity: dict,
    manifest_sha256: str,
    *,
    runs: list[str] | None = None,
    bucket: Bucket | None = None,
) -> None:
    """Refuse a run whose names already stand for something else in an earlier run."""
    _raise(_local_problems(identity, manifest_sha256, runs))
    if bucket is None:
        log.warning("[REGISTRY-LOCAL] no shared bucket for this run, checked local runs only")
        return
    _raise(_shared_problems(identity, manifest_sha256, bucket))


def _local_problems(identity: dict, manifest_sha256: str, runs: list[str] | None) -> list[str]:
    problems = []
    for name in run_names() if runs is None else runs:
        seen = recorded(get_runs_dir() / name)
        for kind in KINDS:
            mine, theirs = identity[kind], seen["identity"].get(kind) or {}
            if theirs.get("ref") != mine["ref"]:
                continue
            if theirs.get("hash") and theirs["hash"] != mine["hash"]:
                problems.append(
                    f"{kind} config {mine['ref']!r} changed since run {name!r} used it "
                    f"(hash {theirs['hash']} -> {mine['hash']})"
                )
            if (
                kind == "dataset"
                and seen["manifest_sha256"]
                and seen["manifest_sha256"] != manifest_sha256
            ):
                problems.append(
                    f"the data behind dataset {mine['ref']!r} changed since run {name!r} "
                    f"(manifest {seen['manifest_sha256'][:12]} -> {manifest_sha256[:12]})"
                )
    return problems


def _shared_problems(identity: dict, manifest_sha256: str, bucket: Bucket) -> list[str]:
    claim_time = bucket.now().isoformat()
    problems = []
    for kind in KINDS:
        mine = identity[kind]
        claim = {
            "ref": mine["ref"],
            "hash": mine["hash"],
            "claimed_by": BenchSettings.load().stity_host,
            "claimed_at": claim_time,
        }
        if kind == "dataset":
            claim["manifest_sha256"] = manifest_sha256
        standing = store.claim_ref(bucket, kind, claim)
        claimant = f"{standing['claimed_by']} at {standing['claimed_at']}"
        if standing["hash"] != mine["hash"]:
            problems.append(
                f"{kind} config {mine['ref']!r} was claimed by {claimant} with another content "
                f"(hash {standing['hash']} -> {mine['hash']})"
            )
        standing_manifest = standing.get("manifest_sha256") or ""
        if kind == "dataset" and standing_manifest != manifest_sha256:
            problems.append(
                f"the data behind dataset {mine['ref']!r} differs from what {claimant} claimed "
                f"(manifest {standing_manifest[:12]} -> {manifest_sha256[:12]})"
            )
    return problems


def _raise(problems: list[str]) -> None:
    if problems:
        raise ConfigError(
            "a config name must keep meaning the same thing. Bump `meta.version` in the "
            "config file (the runs then go to `<name>@v<N>`), or put the old content back:\n"
            + "\n".join(f"  - {p}" for p in dict.fromkeys(problems))
        )
