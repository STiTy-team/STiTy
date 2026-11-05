"""Labels v9x: noise rows on top of the v9d delta corpus - "unintelligible audio is not evidence".

v9d halved the torture-set adapter gap (36 -> 16) but the remaining emissions all sit on faint
scrambled-speech spans, which no training row visits: silence tails teach "quiet means done", nothing
teaches "speech-shaped noise means wait". v9x adds that state. Noise wavs are synthesized from the
aligned training recordings themselves by two techniques chosen to differ from the eval set's 80ms
frame-shuffle (contamination rule - the probe and the training material must not share the generator):

  reversed   one recording, time-reversed, peak scaled to U(0.03, 0.12)
  babble     three recordings summed at random offsets, same scaling

Row families, both with empty deltas (the v9d "emit nothing now" form, qwen3_asr_sft `cond`):

  noise tail   audio|noise.wav#end=u   cond=full committed text  target=""   the live failure site
  noise only   noise.wav#end=u         cond=header only          target=""   noise from cold start

Everything else is train_prefix9d.jsonl carried unchanged, so the single variable vs v9d is the
noise supervision.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix9x.py"""
import argparse
import json
import os
import random

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix9d.jsonl")
ap.add_argument("--noise-dir", default="models/zprobe/data/noise9x")
ap.add_argument("--out", default="models/zprobe/data/train_prefix9x.jsonl")
ap.add_argument("--n-noise", type=int, default=1200, help="noise wavs to synthesize (half reversed, half babble)")
ap.add_argument("--tail-cap", type=int, default=800, help="noise-tail rows per pair (en-ko gets twice this)")
ap.add_argument("--only-per-lang", type=int, default=600, help="noise-only rows per target language")
a = ap.parse_args()
rng = random.Random(11)

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}

by_pair = {}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    by_pair.setdefault((r["lang"], r["tgt"]), []).append(r)
all_recs = [r for recs in by_pair.values() for r in recs]

os.makedirs(a.noise_dir, exist_ok=True)
noise_files = []
for i in range(a.n_noise):
    name = os.path.join(a.noise_dir, f"noise_{i:04d}.wav")
    noise_files.append(os.path.abspath(name))
    if os.path.exists(name):
        continue
    if i % 2 == 0:
        y, sr = sf.read(all_recs[rng.randrange(len(all_recs))]["audio"], dtype="float32")
        y = y[::-1]
    else:
        parts = []
        for r in rng.sample(all_recs, 3):
            p, sr = sf.read(r["audio"], dtype="float32")
            parts.append(p)
        n = max(len(p) for p in parts)
        y = np.zeros(n, dtype="float32")
        for p in parts:
            off = rng.randrange(max(1, n - len(p) + 1))
            y[off:off + len(p)] += p
    span = int(rng.uniform(2.0, 6.0) * sr)
    if len(y) > span:
        off = rng.randrange(len(y) - span + 1)
        y = y[off:off + span]
    peak = float(np.max(np.abs(y))) or 1.0
    y = y * (rng.uniform(0.03, 0.12) / peak)
    sf.write(name, y, sr)
print(f"noise wavs: {len(noise_files)} in {a.noise_dir}", flush=True)


def cum_full(r):
    return " ".join(x.strip() for x in r["adds"] if x.strip())


kept = 0
out = open(a.out, "w", encoding="utf-8")
for l in open(a.base, encoding="utf-8"):
    out.write(l)
    kept += 1

kinds = {"tail": 0, "only": 0}
for (lang, tgt), recs in sorted(by_pair.items()):
    tag = f"language {NAME[tgt]}<asr_text>"
    cap = a.tail_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    for r in rng.sample(recs, min(cap, len(recs))):
        nz = rng.choice(noise_files)
        dur = sf.info(nz).duration
        end = min(rng.uniform(1.0, 2.5), dur)
        out.write(json.dumps({"audio": f"{os.path.abspath(r['audio'])}|{nz}#end={end:.2f}",
                              "cond": tag + cum_full(r), "text": "", "pair": f"{lang}-{tgt}"},
                             ensure_ascii=False) + "\n")
        kinds["tail"] += 1

for tgt in ("ko", "en"):
    tag = f"language {NAME[tgt]}<asr_text>"
    for nz in rng.sample(noise_files, min(a.only_per_lang, len(noise_files))):
        dur = sf.info(nz).duration
        end = min(rng.uniform(1.5, 3.0), dur)
        out.write(json.dumps({"audio": f"{nz}#end={end:.2f}", "cond": tag, "text": "",
                              "pair": f"{tgt}-{tgt}"}, ensure_ascii=False) + "\n")
        kinds["only"] += 1

out.close()
print(f"noise rows: {kinds}", flush=True)
print(f"{a.out}: {kept + sum(kinds.values())} rows ({kept} carried + {sum(kinds.values())} noise)", flush=True)
print("BUILD_PREFIX9X_DONE", flush=True)
