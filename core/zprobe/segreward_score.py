"""Score segreward.py output with COMET-DA (reference) per run; summarise per-boundary validity.
usage: .venv/bin/python segreward_score.py --inp X.jsonl --tgt ko [--eps 0.02]"""
import argparse, json, numpy as np
from comet import download_model, load_from_checkpoint
ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--eps", type=float, default=0.02); ap.add_argument("--out", default="")
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
data, keys = [], []
for r in recs:
    for name, run in r["runs"].items():
        data.append({"src": r["ref_src"], "mt": run["concat"], "ref": r["ref_tgt"]}); keys.append((r["id"], name))
scores = model.predict(data, batch_size=16, gpus=1, progress_bar=False).scores
sc = {k: s for k, s in zip(keys, scores)}
none_s, all_s, single_valid, single_total, deltas = [], [], 0, 0, []
rows = []
for r in recs:
    s0 = sc[(r["id"], "none")]; none_s.append(s0)
    if "all" in r["runs"]: all_s.append(sc[(r["id"], "all")])
    for name, run in r["runs"].items():
        if name.startswith("single"):
            d = sc[(r["id"], name)] - s0; deltas.append(d); single_total += 1; single_valid += d >= -a.eps
            rows.append({"id": r["id"], "bound": run["bounds"][0], "delta": d, "mean_seg_dur": run["mean_seg_dur"]})
print(f"utts={len(recs)}  COMET none={np.mean(none_s):.3f}  all-boundaries={np.mean(all_s) if all_s else float('nan'):.3f}")
print(f"single boundaries: {single_total}; valid (COMET drop <= {a.eps}): {single_valid / max(1, single_total):.2f}; "
      f"mean delta {np.mean(deltas):+.3f}; quartiles {np.percentile(deltas, [25, 50, 75]).round(3).tolist()}")
if a.out:
    json.dump({"scores": {f"{k[0]}|{k[1]}": v for k, v in sc.items()}, "singles": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
