"""Paired COMET delta between two arms scored by comet_eval.py --per_sent (same ids), with a bootstrap 95% interval.
usage: python paired_delta.py --scores DIR --a big2 --b bigcv [--pairs en-ko ...]"""
import argparse, json
import numpy as np

PAIRS = ["en-ko", "en-ja", "en-zh", "ko-en", "ko-ja", "ko-zh", "ja-en", "ja-ko", "ja-zh", "zh-en", "zh-ko", "zh-ja"]
HOLDOUT = {"zh-ko", "ja-ko", "zh-ja"}
ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--pairs", nargs="+", default=PAIRS)
a = ap.parse_args()
rng = np.random.default_rng(0); alld = []


def ci(d):
    m = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


print(f"| pair | {a.a} | {a.b} | {a.b} − {a.a} [95%] |\n|---|---|---|---|")
for p in a.pairs:
    x = {r["id"]: r["comet"] for r in map(json.loads, open(f"{a.scores}/{a.a}_{p}.jsonl"))}
    y = {r["id"]: r["comet"] for r in map(json.loads, open(f"{a.scores}/{a.b}_{p}.jsonl"))}
    d = np.array([y[i] - x[i] for i in x]); alld.append(d); lo, hi = ci(d)
    tag = f"{p} (holdout)" if p in HOLDOUT else p
    print(f"| {tag} | {np.mean(list(x.values())):.3f} | {np.mean(list(y.values())):.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
d = np.concatenate(alld); lo, hi = ci(d)
print(f"| **all** | | | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
