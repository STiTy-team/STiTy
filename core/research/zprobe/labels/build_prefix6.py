"""Labels v6 = v5 plus the full FLEURS Korean train split (2,307 recordings; the prefix3 pipeline kept only
646 of them). ko verbatim stayed the weak spot after v5 (passthrough WER 0.68-0.75) and ko->en is still the
thinnest direction, and FLEURS is n-way parallel, so the gold English text of each Korean sentence is free:
no LLM, no aligner, and no eval contamination (the bench uses the test split; AI Hub sets are eval-only).

Adds, all full-audio rows (streaming shape comes from the v5 prefix/cut rows):
  verbatim ko          audio -> Korean transcript (recordings not already covered by v5's 646)
  ko->en gold          audio -> gold English translation
  ko-ko <SEG> pairs    A_transcript <SEG> [B_transcript]   (boundary row cuts B at 0.8 s)
  ko-en <SEG> pairs    A_gold_en <SEG> [B_gold_en]

usage: python core/research/zprobe/labels/build_prefix6.py --out models/zprobe/data/train_prefix6.jsonl"""
import argparse, csv, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--fleurs", default="/home/mobility/datasets/fleurs/data")
ap.add_argument("--base", default="models/zprobe/data/train_prefix5.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix6.jsonl")
ap.add_argument("--max_dur", type=float, default=24.0)
ap.add_argument("--b_cut_sec", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)


def tsv(path):
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 3:
                yield row


en_text = {r[0]: r[2].strip() for r in tsv(os.path.join(a.fleurs, "en_us", "train.tsv"))}

ko_dir = os.path.join(a.fleurs, "ko_kr")
recs = []
for r in tsv(os.path.join(ko_dir, "train.tsv")):
    wav = os.path.join(ko_dir, "audio", "train", r[1])
    if not os.path.exists(wav):
        continue
    import soundfile as sf
    dur = sf.info(wav).duration
    recs.append({"sid": r[0], "audio": wav, "ko": r[2].strip(), "en": en_text.get(r[0], ""), "dur": dur})

base_rows = [l.rstrip("\n") for l in open(a.base, encoding="utf-8")]
seen_verbatim = {json.loads(l)["audio"] for l in base_rows
                 if (x := json.loads(l))["pair"] == "ko-ko" and "|" not in x["audio"] and "#" not in x["audio"]}

out, counts = [], {}


def add(kind, row):
    out.append(row)
    counts[kind] = counts.get(kind, 0) + 1


KO = "language Korean<asr_text>"
EN = "language English<asr_text>"
for r in recs:
    if r["audio"] not in seen_verbatim:
        add("verbatim_ko", {"audio": r["audio"], "text": KO + r["ko"], "pair": "ko-ko"})
    if r["en"]:
        add("koen_gold", {"audio": r["audio"], "text": EN + r["en"], "pair": "ko-en"})


def clip(r, t):
    return f"{r['audio']}#end={t:.3f}" if t < r["dur"] - 0.05 else r["audio"]


for kind, head, key in (("pair_koko", KO, "ko"), ("pair_koen", EN, "en")):
    g = [r for r in recs if r[key]]
    g.sort(key=lambda r: r["dur"])
    for A, B in zip(g[::2], g[1::2]):
        if A["dur"] + B["dur"] > a.max_dur:
            continue
        add(kind, {"audio": f"{A['audio']}|{clip(B, a.b_cut_sec)}",
                   "text": f"{head}{A[key]} <SEG>", "pair": f"ko-{key}"})
        add(kind, {"audio": f"{A['audio']}|{B['audio']}",
                   "text": f"{head}{A[key]} <SEG> {B[key]}", "pair": f"ko-{key}"})

with open(a.out, "w", encoding="utf-8") as f:
    for l in base_rows:
        f.write(l + "\n")
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{a.out}: {len(base_rows)} base + {len(out)} new")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
