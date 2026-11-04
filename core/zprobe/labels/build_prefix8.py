"""Labels v8 = v7 plus the two northstar gaps that survived the decode fixes: conversational short-utterance
Korean (KsponSpeech part 1 - interjections, backchannels, silence-adjacent speech, so "nothing to say" and
"short in, short out" exist in the training distribution), and mid-phrase cut rows for the x->en directions
(v7 added 15,680 of them for en->ko only, and en regressed -0.019).

The KsponSpeech rows reuse the SEG-labelled jsonl built for seg-mix training
(Qwen3-ASR/finetuning/data/KSponSpeech/train_and_val.jsonl, 10k rows, "language Korean<asr_text>... <SEG> ...");
only the audio path moves to this machine: the referenced pcm (s16le 16k mono) is converted to wav once,
under --kspon-wav. The bench's KsponSpeech evaluation uses the separate eval zip, so nothing leaks.

Third gap, added after the penalty-GRPO probe: silence-tail rows. Every training window so far ends inside
speech, so the state where the news hallucinations fire (utterance over, window still sliding) is absent from
the distribution - and the aligned recordings end exactly at the last utterance (median trailing margin 0.0 s),
so the tail is appended via the loader's "|" concatenation with one shared quiet-noise wav. Label = the full
cumulative text: speech plus silence means say everything and nothing more.

usage: .venv/bin/python core/zprobe/labels/build_prefix8.py --kspon-root /home/mobility/stity-datasets/ksponspeech/data/train01"""
import argparse, json, os, random, re

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--kspon-jsonl", default="Qwen3-ASR/finetuning/data/KSponSpeech/train_and_val.jsonl")
ap.add_argument("--kspon-root", default="/home/mobility/stity-datasets/ksponspeech/data/train01")
ap.add_argument("--kspon-wav", default="/home/mobility/stity-datasets/ksponspeech/data/wav16k")
ap.add_argument("--kspon-max-sec", type=float, default=10.0)
ap.add_argument("--kspon-cap", type=int, default=8000, help="shortest first, so interjections survive the cap")
ap.add_argument("--aligned", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix7.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix8.jsonl")
ap.add_argument("--extra_mids", type=int, default=2, help="extra mid-phrase rows per x->en recording")
ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
ap.add_argument("--sil-cap", type=int, default=800, help="silence-tail rows per pair (en-ko gets twice this)")
a = ap.parse_args()
rng = random.Random(8)

pcm_index = {}
if a.kspon_cap > 0:
    for root, _, files in os.walk(a.kspon_root):
        for f in files:
            if f.endswith(".pcm"):
                pcm_index[f[:-4]] = os.path.join(root, f)
    print(f"pcm indexed: {len(pcm_index)}", flush=True)

os.makedirs(a.kspon_wav, exist_ok=True)
rows_kspon = []
for l in open(a.kspon_jsonl, encoding="utf-8"):
    r = json.loads(l)
    name = os.path.basename(r["audio"]).replace(".wav", "")
    src = pcm_index.get(name)
    if not src:
        continue
    dur = os.path.getsize(src) / 2 / 16000
    if dur > a.kspon_max_sec:
        continue
    rows_kspon.append((dur, name, src, r["text"]))
rows_kspon.sort()
rows_kspon = rows_kspon[: a.kspon_cap]

out_rows = []
for dur, name, src, text in rows_kspon:
    wav = os.path.join(a.kspon_wav, name + ".wav")
    if not os.path.exists(wav):
        pcm = np.frombuffer(open(src, "rb").read(), dtype=np.int16)
        sf.write(wav, pcm.astype(np.float32) / 32768.0, 16000, subtype="PCM_16")
    out_rows.append({"audio": wav, "text": text, "pair": "ko-ko"})
if rows_kspon:
    print(f"kspon rows: {len(out_rows)} (dur <= {a.kspon_max_sec}s, shortest {rows_kspon[0][0]:.1f}s"
          f" ~ longest kept {rows_kspon[-1][0]:.1f}s)", flush=True)

mids = 0
with_tag = {"en": "language English<asr_text>"}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    if r["tgt"] != "en" or r["lang"] == "en":
        continue
    adds, ends = r["adds"], r["ends"]
    cum = [" ".join(x.strip() for x in adds[: i + 1] if x.strip()) for i in range(len(adds))]
    spans = [(j, (ends[j - 1] if j else 0.0) + 0.15, ends[j] - 0.15) for j in range(len(adds))]
    spans = [(j, lo, hi) for j, lo, hi in spans if hi - lo >= 0.3 and hi >= 0.8]
    for j, lo, hi in rng.sample(spans, min(a.extra_mids, len(spans))):
        out_rows.append({"audio": f"{os.path.abspath(r['audio'])}#end={rng.uniform(max(lo, 0.6), hi):.3f}",
                         "text": with_tag["en"] + (cum[j - 1] if j else ""),
                         "pair": f"{r['lang']}-en"})
        mids += 1
print(f"x->en extra mid rows: {mids}", flush=True)

if not os.path.exists(a.sil_wav):
    noise = np.random.default_rng(8).normal(0.0, 0.0015, 4 * 16000).astype(np.float32)
    sf.write(a.sil_wav, noise, 16000, subtype="PCM_16")
TAG = {"ko": "language Korean<asr_text>", "en": "language English<asr_text>"}
by_pair = {}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    by_pair.setdefault((r["lang"], r["tgt"]), []).append(r)
sil = 0
for (lang, tgt), rows in sorted(by_pair.items()):
    cap = a.sil_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    for r in rng.sample(rows, min(cap, len(rows))):
        full = " ".join(x.strip() for x in r["adds"] if x.strip())
        out_rows.append({"audio": f"{os.path.abspath(r['audio'])}|{os.path.abspath(a.sil_wav)}#end={rng.uniform(0.8, 2.0):.2f}",
                         "text": TAG[tgt] + full, "pair": f"{lang}-{tgt}"})
        sil += 1
print(f"silence-tail rows: {sil}", flush=True)

en_recs = {r["audio"]: r for r in by_pair.get(("en", "ko"), []) if r.get("src")}
for r in rng.sample(list(en_recs.values()), min(2000, len(en_recs))):
    out_rows.append({"audio": os.path.abspath(r["audio"]), "text": TAG["en"] + r["src"].strip(), "pair": "en-en"})
print(f"en-en verbatim rows: {min(2000, len(en_recs))} (the kspon rows cover ko-ko; the en secondary stream transcribes en speech)", flush=True)

base = [l.rstrip("\n") for l in open(a.base, encoding="utf-8")]
with open(a.out, "w", encoding="utf-8") as f:
    for l in base:
        f.write(l + "\n")
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{a.out}: {len(base)} base + {len(out_rows)} new", flush=True)
print("BUILD_PREFIX8_DONE", flush=True)
