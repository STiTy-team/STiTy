"""Where does the adapter lose to the Hy cascade? Take the sentences with the largest COMET loss (adapter − ASR→Hy)
and sort them into rule-based types, using the Hy output of the same sentence as the yardstick.

  wrong_script : under 90% of letters in the target script (stayed in the source language / mixed languages)
  loop         : the same 8-character window repeats (degenerate repetition)
  omission     : adapter output under 70% of the Hy output's length
  addition     : adapter output over 140% of the Hy output's length
  lexical      : none of the above — same length and language, different content (wrong word, wrong meaning)

usage: python core/zprobe/gap_breakdown.py --hy models/zprobe/data/hy --b big2 [--frac 0.1] [--examples 3]"""
import argparse, collections, json, os, random

PAIRS = ["en-ko", "en-ja", "en-zh", "ko-en", "ko-ja", "ko-zh", "ja-en", "ja-ko", "ja-zh", "zh-en", "zh-ko", "zh-ja"]
OK = {"en": lambda o: o < 0x250, "ko": lambda o: 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F,
      "zh": lambda o: 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF,
      "ja": lambda o: 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF}


def purity(h, tgt):
    letters = [c for c in h if c.isalpha()]
    return sum(OK[tgt](ord(c)) for c in letters) / len(letters) if letters else 0.0


def looping(h):
    if len(h) < 40:
        return False
    g = [h[i:i + 8] for i in range(len(h) - 8)]
    return len(set(g)) / len(g) < 0.5


def kind(b, hy, tgt):
    if purity(b, tgt) < 0.9:
        return "wrong_script"
    if looping(b):
        return "loop"
    r = len(b) / max(1, len(hy))
    return "omission" if r < 0.7 else "addition" if r > 1.4 else "lexical"


ap = argparse.ArgumentParser()
ap.add_argument("--hy", required=True); ap.add_argument("--b", default="big2")
ap.add_argument("--b_files", default="", help="template with {p}; default <data>/ev_big2_{p}.jsonl next to --hy")
ap.add_argument("--frac", type=float, default=0.1); ap.add_argument("--examples", type=int, default=3)
a = ap.parse_args()
bf = a.b_files or os.path.join(os.path.dirname(a.hy.rstrip("/")), "ev_big2_{p}.jsonl")

rows = []
for p in PAIRS:
    tgt = p.split("-")[1]
    sa = {r["id"]: r["comet"] for r in map(json.loads, open(f"{a.hy}/scores/asr_{p}.jsonl"))}
    sb = {r["id"]: r["comet"] for r in map(json.loads, open(f"{a.hy}/scores/{a.b}_{p}.jsonl"))}
    hy = {r["id"]: r for r in map(json.loads, open(f"{a.hy}/hy_asr_{p}.jsonl"))}
    bo = {r["id"]: r for r in map(json.loads, open(bf.format(p=p)))}
    for i in sa:
        rows.append({"pair": p, "id": i, "d": sb[i] - sa[i], "src": hy[i]["ref_src"], "ref": hy[i]["ref_tgt"],
                     "hy": hy[i]["hyp"], "b": bo[i]["hyp"], "kind": kind(bo[i]["hyp"], hy[i]["hyp"], tgt)})

rows.sort(key=lambda r: r["d"])
total = sum(r["d"] for r in rows)
worst = rows[: int(len(rows) * a.frac)]
base = collections.Counter(r["kind"] for r in rows)
print(f"all {len(rows)} sentences, mean gap {total / len(rows):+.4f}; worst {len(worst)} carry "
      f"{sum(r['d'] for r in worst) / total * 100:.0f}% of the gap\n")
print("| type | worst-10% count | share of worst | share of gap | rate in all sentences | mean Δ |\n|---|---|---|---|---|---|")
for k in ["lexical", "omission", "addition", "wrong_script", "loop"]:
    w = [r for r in worst if r["kind"] == k]
    if not w and not base[k]:
        continue
    print(f"| {k} | {len(w)} | {len(w) / len(worst):.2f} | {sum(r['d'] for r in w) / total * 100:.0f}% "
          f"| {base[k] / len(rows):.3f} | {sum(r['d'] for r in w) / max(1, len(w)):+.3f} |")
print("\n| pair | worst-10% count | expected if uniform |\n|---|---|---|")
cp = collections.Counter(r["pair"] for r in worst); ap_ = collections.Counter(r["pair"] for r in rows)
for p in PAIRS:
    print(f"| {p} | {cp[p]} | {ap_[p] * a.frac:.0f} |")
rnd = random.Random(0)
for k in ["lexical", "omission", "addition", "wrong_script", "loop"]:
    w = [r for r in worst if r["kind"] == k]
    for r in rnd.sample(w, min(a.examples, len(w))):
        print(f"\n[{k}] {r['pair']} {r['id']} Δ{r['d']:+.3f}\n  SRC: {r['src']}\n  HY : {r['hy']}\n  B  : {r['b']}")
