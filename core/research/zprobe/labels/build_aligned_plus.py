"""aligned_plus: the prefix7 aligned pool plus the x->en recordings that never made it in.

connective_en ran its pipeline (pieces -> leak check -> forced alignment) over the whole FLEURS
x->en manifest, but the prefix3 corpus build took roughly half of the aligned output and that cut
carried forward through prefix7 unchanged - the left-out rows are not rejects (556 of the 570
missing ko sids passed the leak check; the used half even contains 49 leak-fails). This script
appends every left-out recording whose sentence passed the leak check, mapping pieces -> adds,
so the en-target delta supervision roughly doubles without any new labelling or alignment.

usage: .venv/bin/python core/research/zprobe/labels/build_aligned_plus.py"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--extra", default="models/zprobe/data/connective_en/aligned.jsonl")
ap.add_argument("--pieces", default="models/zprobe/data/connective_en/pieces1.jsonl")
ap.add_argument("--out", default="models/zprobe/data/prefix7/aligned_plus.jsonl")
a = ap.parse_args()

leak_ok = set()
for l in open(a.pieces, encoding="utf-8"):
    r = json.loads(l)
    if not r.get("fail"):
        leak_ok.add(r["sid"])

have = set()
out = open(a.out, "w", encoding="utf-8")
n_base = 0
for l in open(a.base, encoding="utf-8"):
    out.write(l)
    r = json.loads(l)
    have.add((r["sid"], r["audio"]))
    n_base += 1

added = {"ko": 0, "ja": 0, "zh": 0}
skipped = 0
for l in open(a.extra, encoding="utf-8"):
    r = json.loads(l)
    if r["sid"] not in leak_ok or (r["sid"], r["audio"]) in have:
        skipped += 1
        continue
    if len(r["pieces"]) != len(r["ends"]) or not any(p.strip() for p in r["pieces"]):
        skipped += 1
        continue
    out.write(json.dumps({"id": r["id"], "sid": r["sid"], "audio": r["audio"],
                          "src": r["src"], "lang": r["lang"], "tgt": "en",
                          "dur": float(r["dur"]), "adds": r["pieces"], "ends": r["ends"]},
                         ensure_ascii=False) + "\n")
    added[r["lang"]] += 1
out.close()
print(f"base {n_base} + added {added} (skipped {skipped}) -> {a.out}", flush=True)
print("BUILD_ALIGNED_PLUS_DONE", flush=True)
