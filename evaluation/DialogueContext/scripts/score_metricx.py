"""MetricX-24 (reference-based, google/metricx-24-hybrid-large-v2p6) for a DialogueContext run.

Scores every row that score_reference.py scored, and the probe hypotheses in
probe_scores.jsonl when that file exists. MetricX is an error score: 0 is perfect,
25 is worst. The value is stored as is (`metricx`); tables flip the sign.

Needs the official google-research/metricx `metricx24` package on PYTHONPATH:

    PYTHONPATH=.:<metricx checkout> venv-metrics/bin/python \
        evaluation/DialogueContext/scripts/score_metricx.py \
        --run-dir evaluation/DialogueContext/results/<run_id>
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring_common as common

DEFAULT_MODEL = "google/metricx-24-hybrid-large-v2p6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from core.utils.metrics.meaning import metricx24_score

    run_dir = args.run_dir
    reference = {r["job_id"] for r in common.read_jsonl(run_dir / "scores_reference.jsonl")}
    rows = [r for r in common.latest_by_job(common.read_jsonl(run_dir / "translations.jsonl")).values()
            if r["job_id"] in reference]
    probe_path = run_dir / "probe_scores.jsonl"
    probes = common.read_jsonl(probe_path) if probe_path.exists() else []

    triples: dict[str, dict] = {}
    for src, mt, ref in ([(r["current_source"], r["hypothesis"], r["reference"]) for r in rows]
                         + [(p["src"], p["mt"], p["ref"]) for p in probes]):
        key = common.digest(src, mt, ref)
        triples.setdefault(key, {"id": key, "src": src, "mt": mt, "ref": ref})
    print(f"[metricx] {len(rows)} rows + {len(probes)} probes, {len(triples)} unique triples", flush=True)

    started = time.time()
    result = metricx24_score(list(triples.values()), model_name=args.model, batch_size=args.batch_size)
    scores = result["per_item"]
    print(f"[metricx] scored in {time.time() - started:.1f}s", flush=True)

    common.write_jsonl(run_dir / "scores_metricx.jsonl", [
        {"job_id": r["job_id"], "metricx_model": args.model,
         "metricx": scores[common.digest(r["current_source"], r["hypothesis"], r["reference"])]}
        for r in rows])
    if probes:
        for p in probes:
            p["metricx"] = scores[common.digest(p["src"], p["mt"], p["ref"])]
        common.write_jsonl(probe_path, probes)
    (run_dir / "metricx_status.json").write_text(json.dumps({
        "model": args.model, "tokenizer": result["tokenizer"], "at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows), "n_probes": len(probes), "n_triples": len(triples),
        "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
