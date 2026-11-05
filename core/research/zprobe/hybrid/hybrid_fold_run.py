"""Materialize the region-folded FINAL display of a live hybrid run as its own run directory
(`<pipeline>+fold`), the adopted folding rule (word_fold.py v1): token runs matching the asrref
transcript window [t-8, t+2] fold, a single mismatching token between matches folds with its run,
a run folds only with >= 2 matches and >= 4 normalized chars, an emptied commit is dropped.

usage: (STiTy-bench, bench venv) STITY_DATA_ROOT=... PYTHONPATH=.:bench python .../hybrid_fold_run.py \
         --pipeline hybrid.koen+grpoceren-dedup07 --dataset northstar_ko+en-ko"""

import argparse, copy, re, shutil
from dataclasses import asdict

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
ap.add_argument("--dataset", required=True)
a = ap.parse_args()


def norm(text):
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


env.load()
cfg = config.load(a.pipeline, a.dataset)
spec = datasets.load(cfg.dataset, config.get_data_root())
src_dir = cfg.run_dir

ref = {}
for d in read_jsonl(src_dir / "events.jsonl"):
    if d.get("type") == "asrref":
        ref.setdefault(d["item"], []).append((d["decision_audio_sec"], norm(d["original"])))


def region_fold(item, t, text):
    window = "".join(cx for ct, cx in ref.get(item, []) if t - 8.0 <= ct <= t + 2.0)
    if not window:
        return text
    toks = text.split()
    hit = [len(norm(x)) >= 2 and norm(x) in window for x in toks]
    keep = list(hit)
    for i in range(len(toks)):
        if not keep[i] and 0 < i < len(toks) - 1 and hit[i - 1] and hit[i + 1]:
            keep[i] = True
    out, i = [], 0
    while i < len(toks):
        if not keep[i]:
            out.append(toks[i])
            i += 1
            continue
        j = i
        while j < len(toks) and keep[j]:
            j += 1
        run = toks[i:j]
        solid = sum(1 for k in range(i, j) if hit[k])
        if not (solid >= 2 and len(norm("".join(run))) >= 4):
            out.extend(run)
        i = j
    return re.sub(r"^[\s,.!?…]+", "", " ".join(out))


rows, dropped, trimmed = [], 0, 0
for row in read_jsonl(src_dir / "items.jsonl"):
    row = copy.deepcopy(row)
    recs = []
    for r in row["records"]:
        if r.get("type") != "translated":
            recs.append(r)
            continue
        new = region_fold(
            row["id"], float(r.get("decision_audio_sec") or 0.0), r.get("translation") or ""
        )
        if not norm(new):
            dropped += 1
            continue
        if new != (r.get("translation") or ""):
            trimmed += 1
            r = {**r, "translation": new, "original": new}
        recs.append(r)
    row["records"] = recs
    rows.append(score_item(row, cfg))

run_dir = src_dir.parent / f"{a.pipeline}+fold"
run_dir.mkdir(parents=True, exist_ok=True)
write_jsonl(run_dir / "items.jsonl", rows)
shutil.copy(src_dir / "events.jsonl", run_dir / "events.jsonl")
write_comet_inputs(rows, cfg, run_dir / "comet_inputs.jsonl")
report.write_all(
    cfg=cfg,
    dataset=spec,
    scores=score_run(rows, cfg),
    rows=rows,
    status="ok",
    started=clock.now(),
    wall_sec=0.0,
    run_dir=run_dir,
    pacing={"chunk_size_ms": CHUNK_SIZE_MS, "trailing_silence_ms": TRAILING_SILENCE_MS},
)
print(f"region-folded {src_dir} -> {run_dir} (dropped {dropped}, trimmed {trimmed})")
