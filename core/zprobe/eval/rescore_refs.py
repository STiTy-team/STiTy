"""Rescore a finished bench run against the dataset's current references, without re-decoding.

The decode output never depends on the references, but bench embeds them into items.jsonl at run time and
scores from that copy - so when the gold is corrected (northstar r1, datasets commit 8a50c88) the clean move
is: take the old run's rows, swap every reference field from the freshly loaded dataset, and push them back
through bench's own score_row / score_run / report.write_all. Nothing is reimplemented, and the hypothesis
side stays byte-identical, so old vs new scores differ only by the gold correction. events.jsonl is copied
over for the tools that read it (ns_score.py dual-wer). An item whose audio window moved would invalidate its
recorded stream; the script aborts if id/audio/offset/duration do not match.

usage: (STiTy-bench, bench venv) STITY_DATA_ROOT=... PYTHONPATH=. python .../rescore_refs.py \
         --pipeline adapter.prefix5-segreset-dual --dataset northstar_ko+en-ko --src bench/runs/<ds>/<cfg>.r0"""
import argparse, shutil
from dataclasses import asdict
from pathlib import Path

from core.utils import clock, env
from core.utils.json import read_jsonl, write_jsonl

from bench import config, report
from bench import dataset as datasets
from bench.__main__ import CHUNK_SIZE_MS, TRAILING_SILENCE_MS, score_item, score_run, write_comet_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--src", required=True, help="the finished run directory whose records are kept")
a = ap.parse_args()

env.load()
cfg = config.load(a.pipeline, a.dataset)
spec = datasets.load(cfg.dataset, config.get_data_root())
new = {it.id: asdict(it) for it in spec.items}
REF_FIELDS = ("src_lang", "group", "speaker", "offset", "partial",
              "reference", "reference_translations", "reference_alignment", "reference_segmentation")

rows = list(read_jsonl(Path(a.src) / "items.jsonl"))
started = clock.now()
out_rows = []
for row in rows:
    ni = new.get(row["id"])
    assert ni is not None, f"{row['id']} not in the current dataset"
    assert str(ni["audio"]) == str(row["audio"]) and abs((ni["offset"] or 0) - (row["offset"] or 0)) < 1e-6 \
        and abs(ni["duration_sec"] - row["duration_sec"]) < 1e-6, \
        f"{row['id']}: audio window changed, the recorded stream is not valid for it"
    row.update({k: ni[k] for k in REF_FIELDS})
    out_rows.append(score_item(row, cfg))

run_dir = cfg.run_dir
run_dir.mkdir(parents=True, exist_ok=True)
write_jsonl(run_dir / "items.jsonl", out_rows)
shutil.copy(Path(a.src) / "events.jsonl", run_dir / "events.jsonl")
write_comet_inputs(out_rows, cfg, run_dir / "comet_inputs.jsonl")
report.write_all(cfg=cfg, dataset=spec, scores=score_run(out_rows, cfg), rows=out_rows, status="ok",
                 started=started, wall_sec=0.0, run_dir=run_dir,
                 pacing={"chunk_size_ms": CHUNK_SIZE_MS, "trailing_silence_ms": TRAILING_SILENCE_MS})
print(f"rescored {a.src} -> {run_dir} ({len(out_rows)} items)")
