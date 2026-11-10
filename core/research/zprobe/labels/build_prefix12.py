"""prefix12 stage 3: the prefix10 composition (run on the terra-fine aligned file) plus the three
new axes, each with an off switch so a regression can be dissected:

  context    delta rows get "prompt" = the ASR text a live run would have at that decision point -
             source words whose aligned end falls inside committed 2.0s ASR chunks, minus a decode
             lag; a slice uses a stale lag and a slice is dropped empty so the model survives
             missing context (live coverage is ~50% zero).
  reset-tail a sample of delta rows get another recording's last second glued in front of the
             audio, labels unchanged - the post-slot-reset state (SEG keeps 1.0s, LID keeps 1.5s
             of foreign-language tail), teaching "start the new sentence at its head".
  same-lang  ko-ko / en-en streaming rows cut from word timings with the context attached -
             verbatim continuation becomes a copy task when the ASR text is in the prompt.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix12.py
(expects train_base from: build_prefix10.py --aligned models/zprobe/data/prefix12/aligned.jsonl)
"""
import argparse
import json
import math
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--train-base", default="models/zprobe/data/prefix12/train_base.jsonl")
ap.add_argument("--aligned", default="models/zprobe/data/prefix12/aligned.jsonl")
ap.add_argument("--word-times", default="models/zprobe/data/prefix12/word_times.jsonl")
ap.add_argument("--out", default="models/zprobe/data/train_prefix12.jsonl")
ap.add_argument("--chunk-sec", type=float, default=2.0)
ap.add_argument("--lag-sec", type=float, default=0.5)
ap.add_argument("--stale-frac", type=float, default=0.15, help="slice with a 2.5s lag instead")
ap.add_argument("--context-dropout", type=float, default=0.25)
ap.add_argument("--reset-rows", type=int, default=3000)
ap.add_argument("--lid-frac", type=float, default=0.33, help="share of reset rows with a foreign tail")
ap.add_argument("--samelang-per-lang", type=int, default=800)
ap.add_argument("--no-context", action="store_true")
ap.add_argument("--no-reset", action="store_true")
ap.add_argument("--no-samelang", action="store_true")
a = ap.parse_args()
rng = random.Random(12)

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
SPACED = {"en", "ko"}

words_by_path = {}
lang_by_path = {}
dur_by_path = {}
for l in open(a.word_times, encoding="utf-8"):
    r = json.loads(l)
    p = os.path.abspath(r["audio"].split("#")[0])
    words_by_path[p] = r["words"]
    lang_by_path[p] = r["lang"]
    if r["words"]:
        dur_by_path[p] = r["words"][-1]["end"]
src_by_path = {}
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    p = os.path.abspath(r["audio"].split("#")[0])
    dur_by_path[p] = r["dur"]
    lang_by_path.setdefault(p, r["lang"])
    src_by_path[p] = r["src"]


def _span_bounds(path, i0, i1):
    """src char bounds [lo, hi] covering aligner words [i0, i1) - native spacing and punctuation."""
    src = src_by_path.get(path)
    ws = words_by_path.get(path)
    if src is None or not ws:
        return None
    counts = [sum(c.isalnum() for c in w["w"]) for w in ws]
    lo_target = sum(counts[:i0])
    hi_target = sum(counts[:i1])
    if hi_target == 0:
        return None
    lo = hi = None
    seen = 0
    for idx, ch in enumerate(src):
        if not ch.isalnum():
            continue
        seen += 1
        if lo is None and seen == lo_target + 1:
            lo = idx
        if seen == hi_target:
            hi = idx
            break
    return (lo if lo is not None else 0, hi if hi is not None else len(src) - 1)


def _src_span(path, i0, i1):
    b = _span_bounds(path, i0, i1)
    if b is None:
        return "" if i0 == i1 else None
    src = src_by_path[path]
    return src[b[0]:b[1] + 1]


def parse_audio(audio):
    parts = []
    for seg in audio.split("|"):
        p, _, end = seg.partition("#end=")
        p, _, start = p.partition("#start=")
        parts.append((os.path.abspath(p), float(start) if start else 0.0,
                      float(end) if end else dur_by_path.get(os.path.abspath(p))))
    return parts


def frontier_text(parts, lag):
    offset = 0.0
    ends_at = []
    total = 0.0
    for p, s, e in parts:
        if e is None:
            return None
        ends_at.append((p, s, e, offset))
        offset += e - s
    total = offset
    c = math.floor(max(0.0, total - lag) / a.chunk_sec) * a.chunk_sec
    if c <= 0:
        return ""
    pieces = []
    for p, s, e, off in ends_at:
        ws = words_by_path.get(p)
        if not ws:
            continue
        idx = [i for i, w in enumerate(ws)
               if s <= w["end"] <= e and off + (w["end"] - s) <= c]
        if not idx:
            continue
        span = _src_span(p, idx[0], idx[-1] + 1)
        if span:
            pieces.append(span)
    return " ".join(pieces)


out = open(a.out, "w", encoding="utf-8")
stats = {"carried": 0, "delta": 0, "ctx_attached": 0, "ctx_empty_natural": 0,
         "ctx_dropout": 0, "ctx_stale": 0, "reset_seg": 0, "reset_lid": 0, "samelang": 0}

delta_rows = []
for l in open(a.train_base, encoding="utf-8"):
    r = json.loads(l)
    if not r.get("cond"):
        r.setdefault("prompt", "")
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["carried"] += 1
        continue
    stats["delta"] += 1
    prompt = ""
    if not a.no_context:
        parts = parse_audio(r["audio"])
        if all(p in words_by_path or "silence" in p for p, _, _ in parts):
            if rng.random() < a.context_dropout:
                stats["ctx_dropout"] += 1
            else:
                lag = 2.5 if rng.random() < a.stale_frac else a.lag_sec
                if lag != a.lag_sec:
                    stats["ctx_stale"] += 1
                ft = frontier_text(parts, lag)
                if ft:
                    prompt = ft
                    stats["ctx_attached"] += 1
                else:
                    stats["ctx_empty_natural"] += 1
    r["prompt"] = prompt
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    delta_rows.append(r)

if not a.no_reset:
    pool = {}
    for p, lg in lang_by_path.items():
        if p in words_by_path and dur_by_path.get(p, 0) > 2.5:
            pool.setdefault(lg, []).append(p)
    cands = [r for r in delta_rows if "|" not in r["audio"] and
             r["cond"].endswith("<asr_text>") and
             os.path.abspath(r["audio"].split("#")[0]) in words_by_path]
    rng.shuffle(cands)
    for r in cands[: a.reset_rows]:
        own = os.path.abspath(r["audio"].split("#")[0])
        own_lang = lang_by_path[own]
        lid = rng.random() < a.lid_frac
        tail_lang = rng.choice([lg for lg in pool if lg != own_lang]) if lid else own_lang
        tail = rng.choice(pool[tail_lang])
        if tail == own:
            continue
        keep = 1.5 if lid else 1.0
        td = dur_by_path[tail]
        audio = f"{tail}#start={max(0.0, td - keep):.2f}#end={td:.2f}|{r['audio']}"
        prompt = r.get("prompt", "")
        if prompt and not a.no_context:
            ft = frontier_text(parse_audio(audio), a.lag_sec)
            prompt = ft or ""
        out.write(json.dumps({**r, "audio": audio, "prompt": prompt}, ensure_ascii=False) + "\n")
        stats["reset_lid" if lid else "reset_seg"] += 1

if not a.no_samelang:
    for lg in ("ko", "en"):
        paths = [p for p, l2 in lang_by_path.items()
                 if l2 == lg and p in words_by_path and len(words_by_path[p]) >= 8]
        rng.shuffle(paths)
        for p in paths[: a.samelang_per_lang]:
            ws = words_by_path[p]
            src = src_by_path.get(p, "")

            def _eojeol_ok(i):
                if i >= len(ws):
                    return True
                b = _span_bounds(p, 0, i)
                return b is not None and b[1] + 1 < len(src) and not src[b[1] + 1].isalnum()

            for _ in range(8):
                j = rng.randint(2, max(2, len(ws) - 4))
                k = rng.randint(j + 2, min(len(ws), j + 8))
                if _eojeol_ok(j) and _eojeol_ok(k):
                    break
            else:
                continue
            cut = ws[k - 1]["end"] + 0.2
            audio = p if k == len(ws) else f"{p}#end={cut:.3f}"
            head = _src_span(p, 0, j)
            cont = _src_span(p, j, k)
            if not head or not cont:
                continue
            cond = f"language {NAME[lg]}<asr_text>" + head
            text = " " + cont
            prompt = "" if a.no_context else (frontier_text(parse_audio(audio), a.lag_sec) or "")
            out.write(json.dumps({"audio": audio, "cond": cond, "text": text,
                                  "pair": f"{lg}-{lg}", "prompt": prompt}, ensure_ascii=False) + "\n")
            stats["samelang"] += 1

out.close()
tot = stats["carried"] + stats["delta"] + stats["reset_seg"] + stats["reset_lid"] + stats["samelang"]
print(json.dumps(stats, indent=1))
print(f"{a.out}: {tot} rows")
ctx_den = max(1, stats["ctx_attached"] + stats["ctx_empty_natural"] + stats["ctx_dropout"])
print(f"context 분포: 부착 {stats['ctx_attached']/ctx_den:.0%}, 자연 공백 {stats['ctx_empty_natural']/ctx_den:.0%}, "
      f"드랍아웃 {stats['ctx_dropout']/ctx_den:.0%} (목표: 라이브 제로율 ~50%)")
