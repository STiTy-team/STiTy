"""Table for the Hy-MT2 cascade vs translation-adapter comparison (run after comet_eval.py --per_sent).

Arms, all on the ids of ev_big2_<pair>.jsonl:
  gold  = FLEURS gold source transcript -> Hy-MT2-1.8B          (text-MT ceiling)
  asr   = base Qwen3-ASR-1.7B transcript -> Hy-MT2-1.8B         (cascade we run today)
  big2  = Qwen3-ASR-1.7B + big2 LoRA, target-language tag only  (one-model translation)
Per pair: COMET of each arm, B-A delta with a paired-bootstrap 95% interval, chrF, target-script rate,
and per-utterance latency where ms fields exist.
usage: <comet venv>/bin/python hy_vs_adapter_report.py --hy DIR --scores DIR --data DIR [--out X.md]"""
import argparse, json, os, re
import numpy as np
import sacrebleu

PAIRS = ["en-ko", "en-ja", "en-zh", "ko-en", "ko-ja", "ko-zh", "ja-en", "ja-ko", "ja-zh", "zh-en", "zh-ko", "zh-ja"]
HOLDOUT = {"zh-ko", "ja-ko", "zh-ja"}
OK = {"en": lambda o: o < 0x250, "ko": lambda o: 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F,
      "zh": lambda o: 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF,
      "ja": lambda o: 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF}


def purity(h, tgt):
    letters = [c for c in h if c.isalpha()]
    return sum(OK[tgt](ord(c)) for c in letters) / len(letters) if letters else 0.0


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


ap = argparse.ArgumentParser()
ap.add_argument("--hy", required=True); ap.add_argument("--scores", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--out", default="")
ap.add_argument("--b", default="big2", help="label of the adapter arm (score files are <b>_<pair>.jsonl)")
ap.add_argument("--b_files", default="", help="template with {p}; default <data>/ev_big2_{p}.jsonl")
ap.add_argument("--strip_seg", action="store_true", help="remove <SEG> from the adapter output for chrF / script rate")
ap.add_argument("--pairs", nargs="+", default=PAIRS)
a = ap.parse_args()
Bk = a.b
files = {"gold": os.path.join(a.hy, "hy_gold_{p}.jsonl"), "asr": os.path.join(a.hy, "hy_asr_{p}.jsonl"),
         Bk: a.b_files or os.path.join(a.data, "ev_big2_{p}.jsonl")}
clean = (lambda h: re.sub(r"\s*<SEG>\s*", " ", h).strip()) if a.strip_seg else (lambda h: h)
lines = [f"| pair | N | COMET gold→Hy | COMET ASR→Hy (A) | COMET {Bk} (B) | B−A [95%] | ASR loss (gold−A) "
         "| chrF A / B | target script A / B | <SEG>/output B |", "|---|---|---|---|---|---|---|---|---|---|"]
allA, allB = [], []
for p in a.pairs:
    tgt = p.split("-")[1]
    recs = {k: load(f.format(p=p)) for k, f in files.items()}
    sc = {k: {r["id"]: r["comet"] for r in load(os.path.join(a.scores, f"{k}_{p}.jsonl"))} for k in files}
    ids = [r["id"] for r in recs[Bk]]
    assert all([r["id"] for r in recs[k]] == ids for k in files), f"{p}: id order differs between arms"
    c = {k: np.array([sc[k][i] for i in ids]) for k in files}
    d = c[Bk] - c["asr"]; lo, hi = boot(d)
    allA.append(c["asr"]); allB.append(c[Bk])
    hyp = {"asr": [r["hyp"] for r in recs["asr"]], Bk: [clean(r["hyp"]) for r in recs[Bk]]}
    chrf = {k: sacrebleu.corpus_chrf(hyp[k], [[r["ref_tgt"] for r in recs[k]]]).score for k in hyp}
    pur = {k: np.mean([purity(h, tgt) >= 0.9 for h in hyp[k]]) for k in hyp}
    nseg = np.mean([r["hyp"].count("<SEG>") for r in recs[Bk]])
    tag = f"{p} (holdout)" if p in HOLDOUT else p
    lines.append(f"| {tag} | {len(ids)} | {c['gold'].mean():.3f} | {c['asr'].mean():.3f} | {c[Bk].mean():.3f} "
                 f"| {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {c['gold'].mean() - c['asr'].mean():.3f} "
                 f"| {chrf['asr']:.1f} / {chrf[Bk]:.1f} | {pur['asr']:.2f} / {pur[Bk]:.2f} | {nseg:.2f} |")
A, B = np.concatenate(allA), np.concatenate(allB); lo, hi = boot(B - A)
lines.append(f"| **all** | {len(A)} | | {A.mean():.3f} | {B.mean():.3f} | {(B - A).mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | | | | |")

lat = ["", f"| pair | ASR ms (median) | Hy ms (median) | cascade ms (median of sum) | {Bk} ms (median) |", "|---|---|---|---|---|"]
for p in a.pairs:
    brec = load(files[Bk].format(p=p))
    t = os.path.join(a.hy, f"big2_t_{p}.jsonl")
    if "ms" in brec[0]:
        b = {r["id"]: r["ms"] for r in brec}
    elif Bk == "big2" and os.path.exists(t):
        b = {r["id"]: r["ms"] for r in load(t)}
    else:
        continue
    hy = load(os.path.join(a.hy, f"hy_asr_{p}.jsonl"))
    asr_ms = np.array([r["asr_ms"] for r in hy]); hy_ms = np.array([r["ms"] for r in hy])
    lat.append(f"| {p} | {np.median(asr_ms):.0f} | {np.median(hy_ms):.0f} | {np.median(asr_ms + hy_ms):.0f} "
               f"| {np.median([b[r['id']] for r in hy]):.0f} |")
text = "\n".join(lines + (lat if len(lat) > 3 else []))
print(text)
if a.out:
    open(a.out, "w", encoding="utf-8").write(text + "\n")
