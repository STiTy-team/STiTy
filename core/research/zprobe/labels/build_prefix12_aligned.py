"""prefix12 stage 1: the x->en additions of prefix7/aligned.jsonl swapped for the fine-chunk terra
relabel (same chunk split, earliness-fixed prompt, gtx-arbitrated leak gate). en->ko rows pass
through untouched. Recordings whose forced alignment gave a zero-length chunk (two equal ends)
are dropped - a zero-second audio increment puts two labels on one cut.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix12_aligned.py
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--relabel", default="models/zprobe/data/relabel/full_fine_terra.jsonl")
ap.add_argument("--out", default="models/zprobe/data/prefix12/aligned.jsonl")
a = ap.parse_args()

new = {}
for l in open(a.relabel, encoding="utf-8"):
    r = json.loads(l)
    new[r["sid"]] = r["adds"]

swapped = kept = dropped = enko = 0
with open(a.out, "w", encoding="utf-8") as out:
    for l in open(a.aligned, encoding="utf-8"):
        r = json.loads(l)
        if r["tgt"] == "ko":
            out.write(l)
            enko += 1
            continue
        e = r["ends"]
        if any(b <= a_ for a_, b in zip(e, e[1:])):
            dropped += 1
            continue
        adds = new.get(r["sid"])
        if adds and len(adds) == len(r["ends"]):
            r["adds"] = adds
            swapped += 1
        else:
            kept += 1
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"aligned: en->ko {enko} | x->en terra 교체 {swapped}, 기존 유지 {kept}, 0초 청크 제외 {dropped}")
