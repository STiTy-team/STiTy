"""COMET (Unbabel/wmt22-comet-da, reference-based) over tagswap/steer jsonl files.
Scores ALL hypotheses (wrong-language output is penalized by COMET itself), on the first N ids.
usage: <comet venv>/bin/python comet_eval.py [--strip_seg] [--per_sent DIR] --n 60 LABEL=path.jsonl [...]
--n 0 scores every line. --per_sent writes DIR/LABEL.jsonl ({id, comet} per line) for paired comparison."""
import argparse, json, os, re, sys
from comet import download_model, load_from_checkpoint
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("items", nargs="+")
ap.add_argument("--strip_seg", action="store_true", help="remove <SEG> from hypotheses before scoring; raw scoring penalises SEG-dense outputs")
ap.add_argument("--per_sent", default="", help="directory for per-sentence scores")
ap.add_argument("--kiwi", action="store_true", help="reference-free Unbabel/wmt22-cometkiwi-da instead of wmt22-comet-da")
a = ap.parse_args()
if a.per_sent:
    os.makedirs(a.per_sent, exist_ok=True)
model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da" if a.kiwi else "Unbabel/wmt22-comet-da"))
rows = []
for it in a.items:
    label, path = it.split("=", 1)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = recs[: a.n] if a.n else recs
    hyp = (lambda h: re.sub(r"\s*<SEG>\s*", " ", h).strip()) if a.strip_seg else (lambda h: h)
    data = [{"src": r["ref_src"], "mt": hyp(r["hyp"])} | ({} if a.kiwi else {"ref": r["ref_tgt"]}) for r in recs]
    out = model.predict(data, batch_size=16, gpus=1, progress_bar=False)
    if a.per_sent:
        with open(os.path.join(a.per_sent, f"{label}.jsonl"), "w", encoding="utf-8") as f:
            for r, s in zip(recs, out.scores):
                f.write(json.dumps({"id": r["id"], "comet": s}) + "\n")
    rows.append((label, len(data), out.system_score))
print("\n| run | N | COMET |\n|---|---|---|")
for label, n, s in rows:
    print(f"| {label} | {n} | {s:.3f} |")
