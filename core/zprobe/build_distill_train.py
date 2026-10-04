"""Replace the en-<tgt> segmt rows of a seg9-style train jsonl with distilled rows (distill_seg.py output),
rewriting audio paths to this machine's FLEURS root, keeping only audio/text/pair and skipping\ndistilled rows whose translation degenerated into a repetition loop. Also copies the val jsonl with the same path fix.
usage: build_distill_train.py --base train_seg9.jsonl --distill train_distill_ko_rows.jsonl --tgt ko --out train_distill_ko.jsonl [--val_in val_seg9.jsonl --val_out val_distill_ko.jsonl]"""
import argparse, json, os, re
from collections import Counter
FLEURS = os.path.expanduser("~/datasets/fleurs/data")
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--distill", required=True); ap.add_argument("--tgt", default="ko")
ap.add_argument("--out", required=True); ap.add_argument("--val_in", default=""); ap.add_argument("--val_out", default="")
a = ap.parse_args()
KEEP = ("audio", "text", "pair")
def fix(r):
    r["audio"] = re.sub(r"^.*?/datasets/fleurs/data/", FLEURS + "/", r["audio"])
    return {k: r[k] for k in KEEP}          # one schema for every row; the json loader rejects mixed extra fields
def degenerate(text):
    for seg in (s.strip() for s in text.split("<asr_text>", 1)[1].split("<SEG>")):
        toks = seg.split()
        if len(seg) > 300 or "\ufffd" in seg: return True
        if len(toks) >= 8 and max(Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2)).values()) >= 3: return True
    return False
pair = f"en-{a.tgt}"; n_drop = 0
with open(a.out, "w", encoding="utf-8") as f:
    for l in open(a.base, encoding="utf-8"):
        r = json.loads(l)
        if r.get("pair") == pair and r.get("segmt"): n_drop += 1; continue
        f.write(json.dumps(fix(r), ensure_ascii=False) + "\n")
    n_add = n_bad = 0
    for l in open(a.distill, encoding="utf-8"):
        r = json.loads(l)
        if degenerate(r["text"]): n_bad += 1; continue        # repetition loops from the prefix translator
        f.write(json.dumps(fix(r), ensure_ascii=False) + "\n"); n_add += 1
print(f"{a.out}: dropped {n_drop} {pair} segmt rows, added {n_add} distilled rows, skipped {n_bad} degenerate")
if a.val_in:
    with open(a.val_out, "w", encoding="utf-8") as f:
        for l in open(a.val_in, encoding="utf-8"): f.write(json.dumps(fix(json.loads(l)), ensure_ascii=False) + "\n")
    print(a.val_out)
