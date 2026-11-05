"""Labels v9d: delta rows - the label is only what this step adds, conditioned on what is already committed.

v8b trains (audio prefix -> full cumulative text) while inference feeds the committed text and asks for a
continuation; v9w taped a <WAIT> terminal onto every row and made silence an absorbing state (northstar ko
0.748 -> 0.680, utterances swallowed after one fragment). v9d removes the mismatch instead: a row is
(audio[0:t], cond = header + committed text, target = the words this step adds, or nothing). The committed
part moves into the loss mask (qwen3_asr_sft `cond` field), so "emit nothing now" is supervised directly -
over mid-phrase speech (evidence short), over silence tails (utterance over) - without any new token, and
the decoder needs no wait_token stripping: an empty delta is an immediate EOS, already a legal no-emission step.

Row families:
  boundary delta   audio#end=ends[i]+pad   cond=cum[i-1]  target=adds[i]   (empty adds give empty deltas)
  mid-phrase       audio cut inside phrase  cond=cum[j-1]  target=""        evidence-short silence
  silence tail     audio|quiet#end=u        cond=cum[-1]   target=""        the hallucination site
  seg boundary     A_full|B#end=mid(B[0])   cond=A_full    target=" <SEG>"  sentence done, keep listening
  carried v8b      every row that is not a single-audio #end= cut or a silence tail: plain fulls, kspon,
                   verbatim, concat <SEG> pairs - catchup-from-empty and ASR skills stay cumulative.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix9d.py"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix7/aligned.jsonl")
ap.add_argument("--base", default="models/zprobe/data/train_prefix8b.jsonl")
ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
ap.add_argument("--val", default="models/zprobe/data/val_big_local.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix9d.jsonl")
ap.add_argument("--out-val", default="models/zprobe/data/val_prefix9d.jsonl")
ap.add_argument("--pad", type=float, default=0.2)
ap.add_argument("--sil-cap", type=int, default=800, help="silence-tail rows per pair (en-ko gets twice this)")
ap.add_argument("--seg-cap", type=int, default=2500, help="seg-boundary rows per pair (en-ko gets twice this)")
a = ap.parse_args()
rng = random.Random(9)

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

kinds = {"boundary": 0, "boundary_empty": 0, "mid": 0, "sil": 0, "seg": 0}
by_pair = {}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    by_pair.setdefault((r["lang"], r["tgt"]), []).append(r)


def cum_at(adds, i):
    return " ".join(x.strip() for x in adds[: i + 1] if x.strip())


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
        spans = [(j, (ends[j - 1] if j else 0.0) + 0.15, ends[j] - 0.15) for j in range(len(adds))]
        spans = [(j, lo, hi) for j, lo, hi in spans if hi - lo >= 0.3 and hi >= 0.8]
        if spans:
            j, lo, hi = rng.choice(spans)
            row(f"{path}#end={rng.uniform(max(lo, 0.6), hi):.3f}",
                tag + (cum_at(adds, j - 1) if j else ""), "", pair)
            kinds["mid"] += 1

    sil_wav = os.path.abspath(a.sil_wav)
    cap = a.sil_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    for r in rng.sample(recs, min(cap, len(recs))):
        row(f"{os.path.abspath(r['audio'])}|{sil_wav}#end={rng.uniform(0.8, 2.0):.2f}",
            tag + cum_at(r["adds"], len(r["adds"]) - 1), "", pair)
        kinds["sil"] += 1

    cap = a.seg_cap * (2 if (lang, tgt) == ("en", "ko") else 1)
    pool = [r for r in recs if cum_at(r["adds"], len(r["adds"]) - 1)]
    rng.shuffle(pool)
    for A, B in list(zip(pool[0::2], pool[1::2]))[:cap]:
        t_b = max(0.3, B["ends"][0] * 0.5)
        row(f"{os.path.abspath(A['audio'])}|{os.path.abspath(B['audio'])}#end={t_b:.3f}",
            tag + cum_at(A["adds"], len(A["adds"]) - 1), " <SEG>", pair)
        kinds["seg"] += 1

out.close()
with open(a.out_val, "w", encoding="utf-8") as f:
    for l in open(a.val, encoding="utf-8"):
        r = json.loads(l)
        r["cond"] = ""
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
total = kept + sum(kinds.values())
print(f"delta rows: {kinds}", flush=True)
print(f"{a.out}: {total} rows ({kept} carried + {sum(kinds.values())} delta)", flush=True)
print("BUILD_PREFIX9D_DONE", flush=True)
