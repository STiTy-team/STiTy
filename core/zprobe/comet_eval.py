"""COMET (Unbabel/wmt22-comet-da, reference-based) over tagswap/steer jsonl files.
Scores ALL hypotheses (wrong-language output is penalized by COMET itself), on the first N ids.
usage: <comet venv>/bin/python comet_eval.py [--strip_seg] --n 60 LABEL=path.jsonl [LABEL=path.jsonl ...]"""
import argparse, json, re, sys
from comet import download_model, load_from_checkpoint
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("items", nargs="+")
ap.add_argument("--strip_seg", action="store_true", help="remove <SEG> from hypotheses before scoring; raw scoring penalises SEG-dense outputs")
a = ap.parse_args()
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
rows = []
for it in a.items:
    label, path = it.split("=", 1)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")][: a.n]
    hyp = (lambda h: re.sub(r"\s*<SEG>\s*", " ", h).strip()) if a.strip_seg else (lambda h: h)
    data = [{"src": r["ref_src"], "mt": hyp(r["hyp"]), "ref": r["ref_tgt"]} for r in recs]
    out = model.predict(data, batch_size=16, gpus=1, progress_bar=False)
    rows.append((label, len(data), out.system_score))
print("\n| run | N | COMET |\n|---|---|---|")
for label, n, s in rows:
    print(f"| {label} | {n} | {s:.3f} |")
