"""Every run there is: the ones shared in S3 by any machine, and the ones that only
exist under this computer's bench/runs. Feeds the Runs page."""

from concurrent.futures import ThreadPoolExecutor

from core.integrations.s3 import Bucket

from ... import registry, store
from ...config import get_runs_dir
from ...registry import read_summary
from . import runs

SUMMARY_FETCHERS = 8
FAILURE_CHARS = 300


def host_of(run_id: str) -> str:
    """Run ids are `<stamp>-<host>`; host names may hold dashes, stamps never do."""
    return run_id.partition("-")[2]


def _row(dataset: str, pipeline: str, summary: dict) -> dict:
    identity = summary.get("identity") or {}
    pipeline_meta = identity.get("pipeline") or {}
    dataset_meta = identity.get("dataset") or {}
    counts = summary.get("counts") or {}
    return {
        "dataset": dataset,
        "pipeline": pipeline,
        "stamp": summary.get("stamp") or "",
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "status": summary.get("status") or ("running" if not summary else ""),
        "failure": (summary.get("failure") or "")[:FAILURE_CHARS] or None,
        "items": counts.get("items"),
        "failed_items": len(summary.get("failed_items") or []),
        "wall_sec": counts.get("wall_sec"),
        "commit": (summary.get("source") or {}).get("commit"),
        "models": runs._models(summary),
        "pipeline_description": pipeline_meta.get("description") or "",
        "pipeline_tags": list(pipeline_meta.get("tags") or []),
        "dataset_description": dataset_meta.get("description") or "",
        "dataset_tags": list(dataset_meta.get("tags") or []),
        "metrics": {k: v for k, v in (summary.get("metrics") or {}).items() if isinstance(v, (int, float))},
    }


def _remote_rows(bucket: Bucket, pulled: dict[str, str]) -> tuple[list[dict], set[str]]:
    """The shared runs, and the local run names that are pulled copies of one of them."""
    shared = store.list_runs(bucket)
    with ThreadPoolExecutor(SUMMARY_FETCHERS) as pool:
        summaries = list(pool.map(lambda run: store.summary(bucket, run), shared))
    rows, matched = [], set()
    for run, summary in zip(shared, summaries):
        local_name = f"{run.dataset}/{run.pipeline}"
        is_pulled = pulled.get(local_name) == run.run_id
        if is_pulled:
            matched.add(local_name)
        rows.append(
            {
                **_row(run.dataset, run.pipeline, summary),
                "key": run.folder.rstrip("/"),
                "run_id": run.run_id,
                "host": host_of(run.run_id),
                "local_name": local_name if is_pulled else None,
            }
        )
    return rows, matched


def _local_rows(names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        dataset, pipeline = registry.split(name)
        rows.append(
            {
                **_row(dataset, pipeline, read_summary(get_runs_dir() / name)),
                "key": f"local/{name}",
                "run_id": None,
                "host": None,
                "local_name": name,
            }
        )
    return rows


def all_runs(bucket: Bucket | None) -> dict:
    """Newest first. A shared run pulled to this computer is listed once, as the shared
    run, with `local_name` set; a local run with no shared copy is listed with no `run_id`."""
    local = runs.list_runs() if get_runs_dir().is_dir() else []
    pulled = {name: run_id for name in local if (run_id := store.shared_run_id(get_runs_dir() / name))}
    rows, matched = _remote_rows(bucket, pulled) if bucket is not None else ([], set())
    rows += _local_rows([name for name in local if name not in matched])
    rows.sort(key=lambda row: row["stamp"], reverse=True)
    return {"runs": rows, "shared": bucket is not None}
