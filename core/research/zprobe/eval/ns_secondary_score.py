"""Score the OTHER language of a dual-decode northstar run, from its event log, with bench's own scorers.

A dual run decodes both headers every step but bench scores only the primary (the run's target); the
secondary stream sits in events.jsonl as `dual` records. This rebuilds each item's records as if the
secondary stream were the pipeline output (translated records, translation = the emitted text) and pushes
them through score_row / score_run / write_comet_inputs against the OTHER dataset config, into its own run
directory `<cfg>+from-<src target>`. Caveat: seg resets and the stall-flush clock follow the primary
stream, so this is the dual system's secondary output, not a re-run with the other primary.

usage: (STiTy-bench, bench venv) STITY_DATA_ROOT=... PYTHONPATH=.:bench python .../ns_secondary_score.py \
         --pipeline adapter.prefix7-segreset-dual-flush3 --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en"""

import argparse, copy, shutil
from dataclasses import asdict
from pathlib import Path

from core.utils import clock, env
from core.utils.json import read_jsonl, write_jsonl

from bench import config, report
from bench import dataset as datasets
from bench.__main__ import (
    CHUNK_SIZE_MS,
    TRAILING_SILENCE_MS,
    score_item,
    score_run,
    write_comet_inputs,
)

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--src-dataset", required=True)
ap.add_argument("--dst-dataset", required=True)
a = ap.parse_args()

env.load()
src_cfg = config.load(a.pipeline, a.src_dataset)
dst_cfg = config.load(a.pipeline, a.dst_dataset)
spec = datasets.load(dst_cfg.dataset, config.get_data_root())
new = {it.id: asdict(it) for it in spec.items}
target = dst_cfg.target

src_dir = src_cfg.run_dir
dual = {}
for d in read_jsonl(src_dir / "events.jsonl"):
    if d.get("type") == "dual":
        dual.setdefault(d["item"], []).append(d)

REF_FIELDS = (
    "src_lang",
    "group",
    "speaker",
    "offset",
    "partial",
    "reference",
    "reference_translations",
    "reference_alignment",
    "reference_segmentation",
)
rows = []
for row in read_jsonl(src_dir / "items.jsonl"):
    row = copy.deepcopy(row)
    ni = new[row["id"]]
    row.update({k: ni[k] for k in REF_FIELDS})
    recs = []
    for d in dual.get(row["id"], []):
        recs.append(
            {
                "t": d.get("t"),
                "type": "translated",
                "original": d["original"],
                "translation": d["original"],
                "language": target,
                "target_lang": target,
                "commit_reason": "prefix",
                "decision_audio_sec": d["decision_audio_sec"],
                "committed_elapsed_sec": d.get("committed_elapsed_sec"),
                "translated_elapsed_sec": d.get("committed_elapsed_sec"),
            }
        )
    row["records"] = recs
    row["transcription_output"] = " ".join(r["original"].strip() for r in recs).strip()
    rows.append(score_item(row, dst_cfg))

run_dir = dst_cfg.run_dir.parent / f"{a.pipeline}+from-{a.src_dataset.rsplit('-', 1)[-1]}"
run_dir.mkdir(parents=True, exist_ok=True)
write_jsonl(run_dir / "items.jsonl", rows)
shutil.copy(src_dir / "events.jsonl", run_dir / "events.jsonl")
write_comet_inputs(rows, dst_cfg, run_dir / "comet_inputs.jsonl")
report.write_all(
    cfg=dst_cfg,
    dataset=spec,
    scores=score_run(rows, dst_cfg),
    rows=rows,
    status="ok",
    started=clock.now(),
    wall_sec=0.0,
    run_dir=run_dir,
    pacing={"chunk_size_ms": CHUNK_SIZE_MS, "trailing_silence_ms": TRAILING_SILENCE_MS},
)
print(f"secondary-scored {src_dir} -> {run_dir} ({len(rows)} items)")
