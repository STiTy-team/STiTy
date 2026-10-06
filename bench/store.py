import gzip
from dataclasses import dataclass
from pathlib import Path

from core.integrations.s3 import Bucket, Conflict
from core.utils.json import read_json, write_json

from .settings import BenchSettings

SHARED_MARKER = ".remote"
SUMMARY = "summary.json"
DATA_FILES = ("items.jsonl", "events.jsonl", "comet_inputs.jsonl", "comet_scores.jsonl")


@dataclass(frozen=True)
class RemoteRun:
    dataset: str
    pipeline: str
    run_id: str

    @property
    def folder(self) -> str:
        return f"runs/{self.dataset}/{self.pipeline}/{self.run_id}/"


def shared_run_id(run_dir: Path) -> str | None:
    marker = run_dir / SHARED_MARKER
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None


def _mark_shared(run_dir: Path, run: RemoteRun) -> None:
    (run_dir / SHARED_MARKER).write_text(run.run_id + "\n", encoding="utf-8")


def upload_run(bucket: Bucket, run_dir: Path, host_name: str) -> RemoteRun:
    summary = read_json(run_dir / SUMMARY)
    dataset, _, pipeline = summary["name"].partition("/")
    run = RemoteRun(dataset, pipeline, f"{summary['stamp']}-{host_name}")
    for name in DATA_FILES:
        if (run_dir / name).is_file():
            bucket.put_file(run.folder + name + ".gz", run_dir / name, gzipped=True)
    bucket.put_file(run.folder + SUMMARY, run_dir / SUMMARY, if_absent=True)
    _mark_shared(run_dir, run)
    return run


def list_runs(bucket: Bucket) -> list[RemoteRun]:
    runs = []
    for key in bucket.list("runs/"):
        parts = key.split("/")
        if len(parts) == 5 and parts[4] == SUMMARY:
            _, dataset, pipeline, run_id, _ = parts
            runs.append(RemoteRun(dataset, pipeline, run_id))
    return runs


def _summary_cache(run: RemoteRun) -> Path:
    return BenchSettings.load().xdg_cache_home / "stity" / run.folder / SUMMARY


def summary(bucket: Bucket, run: RemoteRun) -> dict:
    cached = _summary_cache(run)
    if cached.is_file():
        return read_json(cached)
    stored = bucket.get_json(run.folder + SUMMARY)
    if stored is None:
        raise FileNotFoundError(f"no run {run.folder} in the bucket")
    cached.parent.mkdir(parents=True, exist_ok=True)
    write_json(cached, stored.data)
    return stored.data


def _holds_unshared_run(run_dir: Path) -> bool:
    has_files = any((run_dir / name).exists() for name in (SUMMARY, *DATA_FILES))
    return has_files and shared_run_id(run_dir) is None


def pull_run(bucket: Bucket, run: RemoteRun, runs_dir: Path) -> Path:
    run_dir = runs_dir / run.dataset / run.pipeline
    if _holds_unshared_run(run_dir):
        raise FileExistsError(f"{run_dir} holds a run that is only on this machine; move it first")
    stored_summary = bucket.get_bytes(run.folder + SUMMARY)
    if stored_summary is None:
        raise FileNotFoundError(f"no run {run.folder} in the bucket")

    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (SUMMARY, SHARED_MARKER, *DATA_FILES):
        (run_dir / name).unlink(missing_ok=True)
    for name in DATA_FILES:
        stored = bucket.get_bytes(run.folder + name + ".gz")
        if stored is not None:
            (run_dir / name).write_bytes(gzip.decompress(stored.data))
    (run_dir / SUMMARY).write_bytes(stored_summary.data)
    _mark_shared(run_dir, run)
    return run_dir


def read_ref(bucket: Bucket, kind: str, ref: str) -> dict | None:
    stored = bucket.get_json(f"refs/{kind}/{ref}.json")
    return None if stored is None else stored.data


def claim_ref(bucket: Bucket, kind: str, claim: dict) -> dict:
    key = f"refs/{kind}/{claim['ref']}.json"
    try:
        bucket.put_json(key, claim, if_absent=True)
        return claim
    except Conflict:
        return bucket.get_json(key).data
