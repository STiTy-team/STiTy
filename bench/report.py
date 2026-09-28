from datetime import datetime
from pathlib import Path

from core.pipeline import describe as describe_pipeline
from core.utils import clock
from core.utils.json import write_json


def write_all(
    *,
    cfg,
    dataset,
    scores,
    rows,
    status,
    started: datetime,
    wall_sec: float,
    run_dir: Path,
    pacing: dict,
    failure: str | None = None,
) -> Path:
    errored = [r for r in rows if r.get("status") != "ok"]
    empty = [
        r
        for r in rows
        if r.get("status") == "ok" and not (r.get("transcription_output") or "").strip()
    ]
    audio_sec = sum(float(r.get("audio_sec") or 0) for r in rows)
    compute_sec = sum(float(r.get("compute_sec") or 0) for r in rows)
    fed_sec = sum(
        float(r.get("audio_sec") or 0) + pacing["trailing_silence_ms"] / 1000.0
        for r in rows
        if r.get("status") == "ok"
    )
    finished = clock.now()
    metrics, unavailable = scores

    payload = {
        "name": cfg.name,
        "stamp": clock.stamp(started),
        "status": status if status != "ok" else ("degraded" if errored else "ok"),
        "failure": failure,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "metrics": metrics,
        "unavailable": unavailable,
        "counts": {
            "items": len(rows),
            "sessions": len({r.get("group") for r in rows if r.get("group")}),
            "errored": len(errored),
            "empty_transcription_output": len(empty),
            "audio_sec": round(audio_sec, 2),
            "wall_sec": round(wall_sec, 2),
            "compute_sec": round(compute_sec, 2),
            "realtime_factor": round(compute_sec / fed_sec, 3) if fed_sec else None,
        },
        "failed_items": [r.get("id") for r in errored + empty],
        "config": cfg.raw,
        "identity": cfg.identity,
        "pacing": pacing,
        "dataset": dataset.provenance(),
        "components": describe_pipeline(cfg),
    }

    out_path = run_dir / "summary.json"
    write_json(out_path, payload)
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
        print(f"  {key:22} {_show(value)}")
    for metric, reason in payload["unavailable"].items():
        print(f"  - {metric} unavailable: {reason}")
    if payload.get("failure"):
        print(f"  ! run failed: {payload['failure']}")
    if payload["failed_items"]:
        print(f"  ! failed: {', '.join(payload['failed_items'][:5])}")
    print(f"  -> {out_path}")
