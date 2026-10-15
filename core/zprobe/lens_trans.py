"""Logit lens on direct speech translation: does the model pass through the English word before writing Korean?

Teacher-forces the adapter's own translation (from a tagswap jsonl) after the audio and the target-language tag,
then reads every layer's hidden state at every output position through the final norm + lm_head.

1. recall: for the English content words of the base-ASR transcript (single-token words only), the best rank
   each word reaches at any output position, per layer. Control = content words of a different sentence, so a
   generic "English shows up in the middle layers" effect is subtracted.
2. cases: for given (id, correct word, sound-alike word) triples, the best rank of each word per layer, and the
   top-5 tokens per layer at the position where the rank of the correct word is best.

usage: PROBE_MODEL=... PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/lens_trans.py --pair en-ko
         --hyps segmix_en-ko.jsonl --asr asr_base_en.jsonl --limit 100 --case 1930:hostages:prisoners ..."""
import argparse, json, re, sys, time
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_z as pz

STOP = set("""the a an and or but of to in on at for with from by as is are was were be been being this that these those
it its they them their he she his her we our you your i not no so than then there here have has had do does did will would
can could should may might must also into over under about after before more most such very just only other some any each
which who whom what when where why how all both one two three""".split())

ap = argparse.ArgumentParser()
ap.add_argument("--pair", default="en-ko")
ap.add_argument("--hyps", required=True, help="tagswap jsonl of THIS model+adapter")
ap.add_argument("--asr", required=True, help="base ASR jsonl of the source language")
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--case", nargs="*", default=[], help="id:correct:soundalike")
ap.add_argument("--at", nargs="*", default=[], help="id:output_word:correct:soundalike — ranks at the single position that "
                "predicts the first token of output_word, i.e. before that word is in the input (no read-back confound)")
ap.add_argument("--skip_recall", action="store_true")
a = ap.parse_args()

src, tgt = a.pair.split("-")
m = pz.load_model()
thinker = m.model.thinker
tok = m.processor.tokenizer
dev = next(thinker.parameters()).device
asr_tok = tok.convert_tokens_to_ids("<asr_text>")
taps = pz.Taps(thinker)
norm, head = thinker.model.norm, thinker.lm_head
rows = pz.read_rows(src)
hyps = [json.loads(l) for l in open(a.hyps, encoding="utf-8")]
asr = {r["id"]: r["hyp"] for r in map(json.loads, open(a.asr, encoding="utf-8"))}


def single_tok(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def content_words(text):
    out = {}
    for w in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        if len(w) >= 4 and w.lower() not in STOP:
            t = single_tok(w)
            if t is not None:
                out[w] = t
    return out


def lens(uid, hyp, want_ids=False):
    wav, _ = librosa.load(str(pz.FLEURS / pz.LANGS[src][0] / "audio" / "test" / rows[uid][1]), sr=16000)
    prompt = m._build_text_prompt(context="", force_language=pz.LANGS[tgt][1]) + hyp
    inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        thinker(**inputs)
    ids_ = inputs["input_ids"][0]
    asr_i = (ids_ == asr_tok).nonzero()[0, 0].item()
    H = torch.stack(taps.h, 0)[:, 0, asr_i:ids_.shape[0] - 1]          # [L+1, P, D] positions predicting output
    with torch.no_grad():
        lg = head(norm(H)).float()                                      # [L+1, P, V]
    return (lg, ids_[asr_i + 1:]) if want_ids else lg


def best_rank(logits, t):
    """best (lowest) rank of token t over positions, per layer; and the argmin position."""
    tv = logits[..., t:t + 1]
    rank = (logits > tv).sum(-1)                                        # [L+1, P]
    r, pos = rank.min(1)
    return r.cpu().numpy(), pos.cpu().numpy()


ids = [r["id"] for r in hyps][: a.limit]
hyp_of = {r["id"]: r["hyp"] for r in hyps}
cw = {i: content_words(asr[i]) for i in ids}
NL = len(thinker.model.layers) + 1
hit, hit_ctl, n_w, n_c = np.zeros(NL), np.zeros(NL), 0, 0
t0 = time.time()
for k, uid in enumerate([] if a.skip_recall else ids):
    lg = lens(uid, hyp_of[uid])
    own = cw[uid]
    other = {w: t for w, t in cw[ids[(k + 7) % len(ids)]].items() if w not in own}
    for t in own.values():
        hit += best_rank(lg, t)[0] < a.topk; n_w += 1
    for t in other.values():
        hit_ctl += best_rank(lg, t)[0] < a.topk; n_c += 1
    if (k + 1) % 20 == 0:
        print(f"{k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

if not a.skip_recall:
  print(f"\n## English source words in the logit lens while writing {tgt} ({len(ids)} utts, top-{a.topk} at any output position)")
  print(f"own words N={n_w}, control words (another sentence) N={n_c}\n")
  print("| layer | own | control | own − control |\n|---|---|---|---|")
  for L in range(NL):
    print(f"| L{L} | {hit[L] / n_w:.3f} | {hit_ctl[L] / n_c:.3f} | {(hit[L] / n_w) - (hit_ctl[L] / n_c):+.3f} |")

for c in a.case:
    uid, good, bad = c.split(":")
    lg = lens(uid, hyp_of[uid])
    print(f"\n## case {uid}: '{good}' (heard in ASR mode) vs '{bad}' (what the translation suggests)")
    print(f"ASR : {asr[uid]}\nOUT : {hyp_of[uid]}\n")
    tg, tb = single_tok(good), single_tok(bad)
    rg, pg = best_rank(lg, tg)
    rb = best_rank(lg, tb)[0] if tb is not None else None
    print(f"| layer | best rank '{good}' | best rank '{bad}' | top-5 at the position where '{good}' ranks best |\n|---|---|---|---|")
    for L in range(0, NL, 2) if NL > 20 else range(NL):
        top = [tok.decode([int(x)]) for x in lg[L, pg[L]].topk(5).indices]
        print(f"| L{L} | {rg[L]} | {'' if rb is None else rb[L]} | {' · '.join(repr(t) for t in top)} |")

for c in a.at:
    uid, word, good, bad = c.split(":")
    hyp = hyp_of[uid]
    lg, out_ids = lens(uid, hyp, want_ids=True)
    start = hyp.index(word)
    j = next(j for j in range(len(out_ids)) if len(tok.decode(out_ids[: j + 1], skip_special_tokens=True)) > start)
    tg, tb = single_tok(good), single_tok(bad)
    print(f"\n## at {uid}: position predicting '{tok.decode(out_ids[j:j + 1])}' (start of '{word}')")
    print(f"input so far: …{tok.decode(out_ids[max(0, j - 8): j], skip_special_tokens=True)!r}\n")
    print(f"| layer | rank '{good}' | rank '{bad}' | top-5 |\n|---|---|---|---|")
    for L in range(NL):
        row = lg[L, j]
        rk = lambda t: int((row > row[t]).sum()) if t is not None else ""
        top = [tok.decode([int(x)]) for x in row.topk(5).indices]
        print(f"| L{L} | {rk(tg)} | {rk(tb)} | {' · '.join(repr(t) for t in top)} |")
