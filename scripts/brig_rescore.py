"""Score a brig run's OTHER target without re-running the audio.

A brig run records every target's translated/folded stream, and northstar rows carry
reference translations for all languages, so the en view of a ko-dataset run is a pure
rescoring: remap the reference fields to the other dataset config and push the same
records through score_item/score_run. The pipeline is deterministic, so this equals a
live run on the other dataset.

usage: (repo root) uv run --project bench python scripts/brig_rescore.py \
         --pipeline brig.koen+grpoceren --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en"""
import argparse
import copy
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.utils import clock, env
from core.utils.json import read_jsonl, write_jsonl

from bench import config, report
from bench import dataset as datasets
from bench.__main__ import score_item, score_run, write_comet_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True)
ap.add_argument("--src-dataset", required=True)
ap.add_argument("--dst-dataset", required=True)
a = ap.parse_args()

env.load()
src_cfg = config.load(a.pipeline, a.src_dataset)
dst_cfg = config.load(a.pipeline, a.dst_dataset)
spec = datasets.load(dst_cfg.dataset, config.get_data_root())
fresh = {it.id: asdict(it) for it in spec.items}

REF_FIELDS = ("src_lang", "group", "speaker", "offset", "partial",
              "reference", "reference_translations", "reference_alignment", "reference_segmentation")
rows = []
for row in read_jsonl(src_cfg.run_dir / "items.jsonl"):
    row = copy.deepcopy(row)
    row.update({k: fresh[row["id"]][k] for k in REF_FIELDS if k in fresh[row["id"]]})
    rows.append(score_item(row, dst_cfg))

run_dir = dst_cfg.run_dir
run_dir.mkdir(parents=True, exist_ok=True)
write_jsonl(run_dir / "items.jsonl", rows)
shutil.copy(src_cfg.run_dir / "events.jsonl", run_dir / "events.jsonl")
write_comet_inputs(rows, dst_cfg, run_dir / "comet_inputs.jsonl")
report.write_all(cfg=dst_cfg, dataset=spec, scores=score_run(rows, dst_cfg), rows=rows, status="ok",
                 started=clock.now(), wall_sec=0.0, run_dir=run_dir,
                 pacing={"chunk_size_ms": 200, "trailing_silence_ms": 4000})
print(f"rescored {src_cfg.run_dir} -> {run_dir} ({len(rows)} items)")
