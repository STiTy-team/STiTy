"""Labels v10: delta rows with sampled lag - catch-up is supervised, not just punctuality.

v9d taught exactly two behaviours per state: at a phrase end with an up-to-date condition, emit that
phrase; anywhere else, emit nothing. Live decoding at a 0.5s cadence is almost always one phrase
behind, and that state - backlog pending, audio mid-next-phrase - matched only the "wait" rows, so
the model deferred (+600ms northstar) and fragmented short utterances at VAD close. Inference can
emit the whole backlog at any step; v10 makes training say so. Each emission row samples a lag L:
the condition stops L phrases short of what the audio already supports and the target is the joined
backlog, cut both at phrase ends (lag 2 = two phrases at once) and mid-phrase (the live state).
Empty deltas remain only where nothing new is supported, so the empty share falls out of the
sampling (v9d fixed it at 22%) - aimed near 12-15%.

Row families per aligned recording (prefix7 forced alignment, 30-40ms end error):

  on-time boundary   audio#end=ends[i]+pad      cond=cum[i-1]    target=adds[i] (or empty)
  catch-up mid       audio cut inside phrase k  cond=cum[k-1-L]  target=adds[k-L..k-1] joined, L in {1,2}
  catch-up end       audio#end=ends[i]+pad      cond=cum[i-2]    target=adds[i-1..i] joined
  wait mid           audio cut inside phrase k  cond=cum[k-1]    target=""   (halved vs v9d)
  silence tail A     audio|quiet#end=u          cond=cum[-1]     target=""   utterance over
  silence tail B     audio|quiet#end=u          cond=cum[-1-L]   target=backlog   catch up at the end
  seg boundary       as v9d                     cond=A_full      target=" <SEG>"
  carried v8b        as v9d: plain fulls, kspon, verbatim, concat <SEG> pairs

usage: .venv/bin/python core/research/zprobe/labels/build_prefix10.py"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix8b.jsonl")
ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
ap.add_argument("--out", default="models/zprobe/data/train_prefix10.jsonl")
ap.add_argument("--pad", type=float, default=0.2)
ap.add_argument("--catchup-per-rec", type=int, default=2)
ap.add_argument("--wait-frac", type=float, default=0.5, help="chance of one wait-mid row per recording")
ap.add_argument("--sil-cap", type=int, default=800, help="silence-tail rows per pair (en-ko gets twice this)")
ap.add_argument("--seg-cap", type=int, default=2500, help="seg-boundary rows per pair (en-ko gets twice this)")
a = ap.parse_args()
rng = random.Random(10)

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}

kept = dropped = 0
out = open(a.out, "w", encoding="utf-8")
for l in open(a.base, encoding="utf-8"):
    audio = json.loads(l)["audio"]
    single_cut = "#end=" in audio and "|" not in audio
    silence = os.path.basename(a.sil_wav) in audio
    if single_cut or silence:
        dropped += 1
        continue
    r = json.loads(l)
    r["cond"] = ""
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept += 1
print(f"base: kept {kept}, dropped {dropped} (single-audio cuts + silence tails)", flush=True)

kinds = {"boundary": 0, "boundary_empty": 0, "catchup_mid": 0, "catchup_end": 0,
         "wait_mid": 0, "sil_end": 0, "sil_catchup": 0, "seg": 0}
by_pair = {}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    by_pair.setdefault((r["lang"], r["tgt"]), []).append(r)


def cum_at(adds, i):
    return " ".join(x.strip() for x in adds[: i + 1] if x.strip())


def backlog(adds, lo, hi):
    return " ".join(x.strip() for x in adds[lo: hi + 1] if x.strip())


def row(audio, cond, delta, pair):
    out.write(json.dumps({"audio": audio, "cond": cond, "text": delta, "pair": pair},
                         ensure_ascii=False) + "\n")


for (lang, tgt), recs in sorted(by_pair.items()):
    tag = f"language {NAME[tgt]}<asr_text>"
    pair = f"{lang}-{tgt}"
    for r in recs:
        adds, ends, path = r["adds"], r["ends"], os.path.abspath(r["audio"])
        last = len(adds) - 1
        for i in range(len(adds)):
            if ends[i] < 0.6:
                continue
            committed = cum_at(adds, i - 1) if i else ""
            add = adds[i].strip()
            delta = (" " + add if committed else add) if add else ""
            audio = path if i == last else f"{path}#end={ends[i] + a.pad:.3f}"
            row(audio, tag + committed, delta, pair)
            kinds["boundary" if delta else "boundary_empty"] += 1

        spans = [(k, (ends[k - 1] if k else 0.0) + 0.15, ends[k] - 0.15) for k in range(len(adds))]
        spans = [(k, lo, hi) for k, lo, hi in spans if hi - lo >= 0.3 and hi >= 0.8]

        catch_spans = [(k, lo, hi) for k, lo, hi in spans if k >= 1 and backlog(adds, 0, k - 1)]
        for _ in range(min(a.catchup_per_rec, len(catch_spans))):
            k, lo, hi = catch_spans.pop(rng.randrange(len(catch_spans)))
            lag = rng.choice((1, 1, 2))
            lag = min(lag, k)
            committed = cum_at(adds, k - 1 - lag)
            delta = backlog(adds, k - lag, k - 1)
            if not delta:
                continue
            row(f"{path}#end={rng.uniform(max(lo, 0.6), hi):.3f}",
                tag + committed, (" " + delta if committed else delta), pair)
            kinds["catchup_mid"] += 1

        ends_ok = [i for i in range(2, len(adds)) if ends[i] >= 0.6 and backlog(adds, i - 1, i)]
        if ends_ok:
            i = rng.choice(ends_ok)
            committed = cum_at(adds, i - 2)
            delta = backlog(adds, i - 1, i)
            audio = path if i == last else f"{path}#end={ends[i] + a.pad:.3f}"
            row(audio, tag + committed, (" " + delta if committed else delta), pair)
            kinds["catchup_end"] += 1

        if spans and rng.random() < a.wait_frac:
            k, lo, hi = rng.choice(spans)
            row(f"{path}#end={rng.uniform(max(lo, 0.6), hi):.3f}",
                tag + (cum_at(adds, k - 1) if k else ""), "", pair)
            kinds["wait_mid"] += 1

    sil_wav = os.path.abspath(a.sil_wav)
    cap = a.sil_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    pool = rng.sample(recs, min(cap, len(recs)))
    for n, r in enumerate(pool):
        adds = r["adds"]
        full = cum_at(adds, len(adds) - 1)
        audio = f"{os.path.abspath(r['audio'])}|{sil_wav}#end={rng.uniform(0.8, 2.0):.2f}"
        if n % 2 == 0 or len([x for x in adds if x.strip()]) < 2:
            row(audio, tag + full, "", pair)
            kinds["sil_end"] += 1
        else:
            lag = min(rng.choice((1, 1, 2)), len(adds) - 1)
            committed = cum_at(adds, len(adds) - 1 - lag)
            delta = backlog(adds, len(adds) - lag, len(adds) - 1)
            if not delta:
                row(audio, tag + full, "", pair)
                kinds["sil_end"] += 1
                continue
            row(audio, tag + committed, (" " + delta if committed else delta), pair)
            kinds["sil_catchup"] += 1

    cap = a.seg_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    pool = [r for r in recs if cum_at(r["adds"], len(r["adds"]) - 1)]
    rng.shuffle(pool)
    for A, B in list(zip(pool[0::2], pool[1::2]))[:cap]:
        t_b = max(0.3, B["ends"][0] * 0.5)
        row(f"{os.path.abspath(A['audio'])}|{os.path.abspath(B['audio'])}#end={t_b:.3f}",
            tag + cum_at(A["adds"], len(A["adds"]) - 1), " <SEG>", pair)
        kinds["seg"] += 1

out.close()
total = kept + sum(kinds.values())
empty = kinds["boundary_empty"] + kinds["wait_mid"] + kinds["sil_end"]
print(f"delta rows: {kinds}", flush=True)
print(f"{a.out}: {total} rows ({kept} carried + {sum(kinds.values())} delta), "
      f"empty deltas {empty} ({empty / max(1, sum(kinds.values())):.0%} of delta rows)", flush=True)
print("BUILD_PREFIX10_DONE", flush=True)
