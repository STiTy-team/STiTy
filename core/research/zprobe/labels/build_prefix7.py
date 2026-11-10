"""Labels v7 = the v6 composition rebuilt on top of the relabeled en->ko additions, plus extra mid-phrase
cut rows. The en->ko adds (Qwen3-8B) are swapped for the gpt-6-luna relabel (natural spoken Korean, strict
no-lookahead, Hy-prefix leak check; rows the relabel failed keep their old adds), then the original builders
rerun unchanged - build_prefix.py build -> build_prefix4 -> build_prefix5 -> build_prefix6 - so every row
construction stays the original code's. The extra mid rows (audio cut inside a phrase, text up to the last
completed addition) sharpen "EOS only at word boundaries" against the mid-eojeol commits seen on northstar.

usage: python core/research/zprobe/labels/build_prefix7.py --relabel models/zprobe/data/relabel/full_enko.jsonl"""

import argparse, json, os, random, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--relabel", default="models/zprobe/data/relabel/full_enko.jsonl")
ap.add_argument("--aligned", default="models/zprobe/data/prefix3/aligned.jsonl")
ap.add_argument("--data", default="models/zprobe/data")
ap.add_argument(
    "--extra_mids", type=int, default=2, help="extra mid-phrase rows per en->ko recording"
)
a = ap.parse_args()
D = a.data
V7 = f"{D}/prefix7"
os.makedirs(V7, exist_ok=True)

new = {json.loads(l)["sid"]: json.loads(l)["adds"] for l in open(a.relabel, encoding="utf-8")}
swapped = kept = 0
with open(f"{V7}/aligned.jsonl", "w", encoding="utf-8") as f:
    for l in open(a.aligned, encoding="utf-8"):
        r = json.loads(l)
        adds = new.get(r["sid"]) if r["lang"] == "en" and r["tgt"] == "ko" else None
        if adds and len(adds) == len(r["ends"]):
            r["adds"] = adds
            swapped += 1
        elif r["lang"] == "en" and r["tgt"] == "ko":
            kept += 1
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"aligned: {swapped} swapped, {kept} kept old", flush=True)


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


run(
    "core/research/zprobe/labels/build_prefix.py",
    "build",
    "--aligned",
    f"{V7}/aligned.jsonl",
    "--base",
    f"{D}/train_seg9_mix.jsonl",
    "--wav_dir",
    f"{V7}/cuts",
    "--out",
    f"{V7}/train_v3.jsonl",
)

rng = random.Random(7)
extra = 0
with open(f"{V7}/train_v3.jsonl", "a", encoding="utf-8") as f:
    for l in open(f"{V7}/aligned.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["lang"] != "en" or r["tgt"] != "ko":
            continue
        adds, ends = r["adds"], r["ends"]
        cum = [" ".join(x.strip() for x in adds[: i + 1] if x.strip()) for i in range(len(adds))]
        spans = [(j, (ends[j - 1] if j else 0.0) + 0.15, ends[j] - 0.15) for j in range(len(adds))]
        spans = [(j, lo, hi) for j, lo, hi in spans if hi - lo >= 0.3 and hi >= 0.8]
        for j, lo, hi in rng.sample(spans, min(a.extra_mids, len(spans))):
            f.write(
                json.dumps(
                    {
                        "audio": f"{os.path.abspath(r['audio'])}#end={rng.uniform(max(lo, 0.6), hi):.3f}",
                        "text": f"language Korean<asr_text>" + (cum[j - 1] if j else ""),
                        "pair": "en-ko",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            extra += 1
print(f"extra mid rows: {extra}", flush=True)

run(
    "core/research/zprobe/labels/build_prefix4.py",
    "--aligned",
    f"{V7}/aligned.jsonl",
    "--base",
    f"{V7}/train_v3.jsonl",
    "--out",
    f"{V7}/train_v4.jsonl",
)
run(
    "core/research/zprobe/labels/build_prefix5.py",
    "--aligned",
    f"{V7}/aligned.jsonl",
    "--base",
    f"{V7}/train_v4.jsonl",
    "--out",
    f"{V7}/train_v5.jsonl",
)
run(
    "core/research/zprobe/labels/build_prefix6.py",
    "--base",
    f"{V7}/train_v5.jsonl",
    "--out",
    f"{D}/train_prefix7.jsonl",
)
for t in ("train_v3", "train_v4", "train_v5"):
    os.remove(f"{V7}/{t}.jsonl")
print("BUILD_PREFIX7_DONE", flush=True)
