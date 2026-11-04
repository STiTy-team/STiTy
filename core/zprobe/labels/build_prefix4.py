"""Labels v4: teach the streaming adapter that a finished sentence is not the end of the stream. Pairs of prefix3
aligned sentences (same direction) are joined into one utterance; the target separates them with <SEG>, so <SEG>
becomes the model's explicit "sentence done, keep listening" token and EOS goes back to meaning only "wait".
No audio copies and no LLM calls: rows reuse the prefix3 alignments, and the audio field joins parts as
"a.wav#end=tA|b.wav#end=tB" (qwen3_asr_sft.load_audio concatenates them).

Two rows per pair, over audio A_full + B[0:t] (a third mid-B row pushed the feature cache past the 32 GB of
/dev/shm and was dropped):
  boundary  t = mid of B's first phrase   -> A_translation + " <SEG>"            (emit the marker, then wait)
  full      t = B's end                   -> A + " <SEG> " + B_translation

usage: python core/zprobe/labels/build_prefix4.py --aligned models/zprobe/data/prefix3/aligned.jsonl \
           --base models/zprobe/data/train_prefix3.jsonl --out models/zprobe/data/train_prefix4.jsonl"""
import argparse, json, random

NAME = {"ko": "Korean", "en": "English"}

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix3/aligned.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix3.jsonl",
                help="existing rows to keep (single-sentence prefix behavior must not be forgotten)")
ap.add_argument("--out", default="models/zprobe/data/train_prefix4.jsonl")
ap.add_argument("--max_dur", type=float, default=18.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)


def text_of(adds):
    return " ".join(x.strip() for x in adds if x.strip())


def clip(r, t):
    return f"{r['audio']}#end={t:.3f}" if t < r["dur"] - 0.05 else r["audio"]


rows = [json.loads(l) for l in open(a.aligned, encoding="utf-8")
        if (x := json.loads(l)) and x["adds"] and x["ends"] and text_of(x["adds"])]
by_dir = {}
for r in rows:
    by_dir.setdefault((r["lang"], r["tgt"]), []).append(r)

out, pairs = [], 0
for key, group in sorted(by_dir.items()):
    rng.shuffle(group)
    for A, B in zip(group[::2], group[1::2]):
        if A["dur"] + B["dur"] > a.max_dur:
            continue
        pairs += 1
        pair = f"{key[0]}-{key[1]}"
        head = f"language {NAME[B['tgt']]}<asr_text>"
        a_txt = text_of(A["adds"])

        t_b = max(0.3, B["ends"][0] * 0.5)
        out.append({"audio": f"{A['audio']}|{clip(B, t_b)}",
                    "text": f"{head}{a_txt} <SEG>", "pair": pair})

        out.append({"audio": f"{A['audio']}|{B['audio']}",
                    "text": f"{head}{a_txt} <SEG> {text_of(B['adds'])}", "pair": pair})

base = [l.rstrip("\n") for l in open(a.base, encoding="utf-8")] if a.base else []
with open(a.out, "w", encoding="utf-8") as f:
    for l in base:
        f.write(l + "\n")
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{a.out}: {len(base)} base + {len(out)} concat rows ({pairs} pairs)")
