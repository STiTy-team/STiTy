"""Extend prefix12/word_times.jsonl with word-level forced alignment for the prefix13 sources:
FLEURS ko/en train (full), kspon conv, DailyTalk conv. Resume by audio path; appends and flushes
per row. Transcripts: FLEURS train.tsv, conv manifests' src.

usage: PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/lid/align_more.py
       [--manifest sentences.jsonl --out word_times.jsonl [--reuse old_word_times.jsonl ...]]

With --manifest the hard-coded sources are skipped and every rec of every sentence is aligned
(id = "<sid>#<n>"). Audio may carry "#start=S#end=E" (seconds); word ends are then relative to S.
--reuse copies rows of an earlier word_times file for whole-file recs instead of re-aligning.
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "Qwen3-ASR", "finetuning"))
from qwen3_asr_sft import load_audio  # noqa: E402
from qwen_asr import Qwen3ForcedAligner  # noqa: E402

OUT = "models/zprobe/data/prefix12/word_times.jsonl"
NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}

recs = []


def add(rid, audio, lang, src):
    recs.append({"id": rid, "audio": audio, "lang": lang, "src": src})


ap = argparse.ArgumentParser()
ap.add_argument("--manifest")
ap.add_argument("--out", default=OUT)
ap.add_argument("--reuse", action="append", default=[])
args = ap.parse_args()
OUT = args.out

if args.manifest:
    for l in open(args.manifest, encoding="utf-8"):
        if not l.strip():
            continue
        sent = json.loads(l)
        for n, rec in enumerate(sent["recs"]):
            add(f"{sent['sid']}#{n}", rec["audio"], sent["lang"], sent["src"])
else:
    for lang, sub in (("ko", "ko_kr"), ("en", "en_us")):
        base = os.path.expanduser(f"~/datasets/fleurs/data/{sub}")
        for row in csv.reader(open(f"{base}/train.tsv", encoding="utf-8"), delimiter="\t"):
            if len(row) < 3:
                continue
            p = f"{base}/audio/train/{row[1]}"
            if os.path.exists(p):
                add(f"fl13_{lang}_{row[1].removesuffix('.wav')}", p, lang, row[2])

    for name, lang in (("kspon_manifest", "ko"), ("dailytalk_manifest", "en")):
        for l in open(f"models/zprobe/data/conv/{name}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            add(r["id"], r["audio"].split("#")[0], lang, r["src"])

if args.manifest:
    # resume key = stored audio string (suffix included)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            done.add(json.loads(l)["audio"])
    key = lambda r: r["audio"]  # noqa: E731
else:
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            done.add(os.path.abspath(json.loads(l)["audio"].split("#")[0]))
    key = lambda r: os.path.abspath(r["audio"])  # noqa: E731
todo = [r for r in recs if key(r) not in done]
seen = set()
todo = [r for r in todo if not (key(r) in seen or seen.add(key(r)))]
print(f"sources {len(recs)}, already aligned {len(recs) - len(todo)}, todo {len(todo)}", flush=True)

reuse_rows = {}
for rp in args.reuse:
    for l in open(rp, encoding="utf-8"):
        row = json.loads(l)
        reuse_rows.setdefault(os.path.abspath(row["audio"].split("#")[0]), row)
reused = 0
if reuse_rows:
    rest = []
    fo = open(OUT, "a", encoding="utf-8")
    for r in todo:
        row = reuse_rows.get(os.path.abspath(r["audio"].split("#")[0])) if "#" not in r["audio"] else None
        if row is None:
            rest.append(r)
            continue
        fo.write(json.dumps({"id": r["id"], "audio": r["audio"], "lang": r["lang"],
                             "words": row["words"]}, ensure_ascii=False) + "\n")
        reused += 1
    fo.close()
    todo = rest
    print(f"reused {reused}, to align {len(todo)}", flush=True)

aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B",
                                             dtype=torch.bfloat16, device_map="cuda")
out = open(OUT, "a", encoding="utf-8")
t0 = time.time()
ok = fail = 0
for n, r in enumerate(todo):
    try:
        wav = load_audio(r["audio"], 16000)
        items = aligner.align((wav, 16000), r["src"], NAME[r["lang"]])[0]
        words = [{"w": it.text, "end": round(float(it.end_time), 2)} for it in items]
    except Exception as e:  # noqa: BLE001
        fail += 1
        if fail <= 5:
            print(f"FAIL {r['id']}: {e}", flush=True)
        continue
    out.write(json.dumps({"id": r["id"], "audio": r["audio"], "lang": r["lang"],
                          "words": words}, ensure_ascii=False) + "\n")
    out.flush()
    ok += 1
    if (n + 1) % 1000 == 0:
        print(f"{n + 1}/{len(todo)} ok={ok} fail={fail} {time.time() - t0:.0f}s", flush=True)
print(f"ALIGN_MORE_DONE ok={ok} fail={fail} reused={reused} {time.time() - t0:.0f}s", flush=True)
