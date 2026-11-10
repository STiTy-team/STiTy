"""prefix13 stage 2: the prefix12 composition plus a source-language tag on every row and a
transcription track sized to the translation track.

  tag           cond (and carried rows' text) gains "speech {Source} " before "language {Target}" -
                the probe LID's verdict at runtime. Always filled, never a '?' state: at inference
                the model cannot know the tag is wrong, so training mirrors that - wrong tags appear
                at the measured rates and places (onset rows 10%, steady 3%, reset-tail rows 20%
                stale = the glued tail's language). Labels always follow the audio: the tag is a
                hint, the ears win.
  transcription ko-ko / en-en streaming rows (cond = transcript prefix, text = continuation, plus
                full rows) cut at eojeol boundaries from word_times, sources FLEURS ko/en + kspon +
                DailyTalk, scaled to --transcribe-ratio x the translation delta rows and balanced
                50:50 ko/en. The single-mast transcription task: speech X language X. NO context on
                these rows - the system-slot ASR text would be the answer's own prefix (label leak),
                and the single-mast target has no ASR text to lean on; transcription is trained as
                pure listening. Translation rows keep the context (a transform task, not a leak).
  context/reset as build_prefix12 (ASR-frontier prompt, reset-tail rows).

expects: prefix13/aligned.jsonl (build_prefix13_aligned.py) and prefix13/train_base.jsonl
         (build_prefix10.py --aligned prefix13/aligned.jsonl).
usage: .venv/bin/python core/research/zprobe/labels/build_prefix13.py
"""
import argparse
import csv
import json
import math
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--train-base", default="models/zprobe/data/prefix13/train_base.jsonl")
ap.add_argument("--aligned", default="models/zprobe/data/prefix13/aligned.jsonl")
ap.add_argument("--word-times", default="models/zprobe/data/prefix12/word_times.jsonl")
ap.add_argument("--conv-dir", default="models/zprobe/data/conv")
ap.add_argument("--fleurs", default="/home/mobility/datasets/fleurs/data")
ap.add_argument("--out", default="models/zprobe/data/train_prefix13.jsonl")
ap.add_argument("--chunk-sec", type=float, default=2.0)
ap.add_argument("--lag-sec", type=float, default=0.5)
ap.add_argument("--stale-frac", type=float, default=0.15)
ap.add_argument("--context-dropout", type=float, default=0.35)
ap.add_argument("--reset-rows", type=int, default=3000)
ap.add_argument("--lid-frac", type=float, default=0.33)
ap.add_argument("--transcribe-ratio", type=float, default=1.0)
ap.add_argument("--rows-per-rec", type=int, default=5)
ap.add_argument("--tag-wrong-onset", type=float, default=0.10)
ap.add_argument("--tag-wrong-steady", type=float, default=0.03)
ap.add_argument("--reset-tag-stale", type=float, default=0.20)
ap.add_argument("--onset-sec", type=float, default=1.2)
ap.add_argument("--no-context", action="store_true")
ap.add_argument("--no-reset", action="store_true")
ap.add_argument("--no-transcribe", action="store_true")
ap.add_argument("--no-tags", action="store_true")
a = ap.parse_args()
rng = random.Random(13)

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
LANGS = list(NAME)

words_by_path = {}
lang_by_path = {}
dur_by_path = {}
src_by_path = {}
for l in open(a.word_times, encoding="utf-8"):
    r = json.loads(l)
    p = os.path.abspath(r["audio"].split("#")[0])
    words_by_path[p] = r["words"]
    lang_by_path[p] = r["lang"]
    if r["words"]:
        dur_by_path[p] = r["words"][-1]["end"]
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    p = os.path.abspath(r["audio"].split("#")[0])
    dur_by_path[p] = r["dur"]
    lang_by_path.setdefault(p, r["lang"])
    src_by_path[p] = r["src"]
for lang, sub in (("ko", "ko_kr"), ("en", "en_us")):
    base = f"{a.fleurs}/{sub}"
    try:
        for row in csv.reader(open(f"{base}/train.tsv", encoding="utf-8"),
                              delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 3:
                src_by_path.setdefault(os.path.abspath(f"{base}/audio/train/{row[1]}"), row[2].strip())
    except FileNotFoundError:
        pass
for name in ("kspon_manifest", "dailytalk_manifest"):
    for l in open(f"{a.conv_dir}/{name}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        src_by_path.setdefault(os.path.abspath(r["audio"].split("#")[0]), r["src"])


def _span_bounds(path, i0, i1):
    src = src_by_path.get(path)
    ws = words_by_path.get(path)
    if src is None or not ws:
        return None
    counts = [sum(c.isalnum() for c in w["w"]) for w in ws]
    lo_target, hi_target = sum(counts[:i0]), sum(counts[:i1])
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
    return src_by_path[path][b[0]:b[1] + 1]


def parse_audio(audio):
    parts = []
    for seg in audio.split("|"):
        p, _, end = seg.partition("#end=")
        p, _, start = p.partition("#start=")
        parts.append((os.path.abspath(p), float(start) if start else 0.0,
                      float(end) if end else dur_by_path.get(os.path.abspath(p))))
    return parts


def total_sec(parts):
    return None if any(e is None for _, _, e in parts) else sum(e - s for _, s, e in parts)


def frontier_text(parts, lag):
    offset = 0.0
    ends_at = []
    for p, s, e in parts:
        if e is None:
            return None
        ends_at.append((p, s, e, offset))
        offset += e - s
    c = math.floor(max(0.0, offset - lag) / a.chunk_sec) * a.chunk_sec
    if c <= 0:
        return ""
    pieces = []
    for p, s, e, off in ends_at:
        ws = words_by_path.get(p)
        if not ws:
            continue
        idx = [i for i, w in enumerate(ws) if s <= w["end"] <= e and off + (w["end"] - s) <= c]
        if not idx:
            continue
        span = _src_span(p, idx[0], idx[-1] + 1)
        if span:
            pieces.append(span)
    return " ".join(pieces)


def tag_of(true_lang, parts, stale_lang=None):
    """The LID verdict this row's moment would carry: mostly right, wrong at the measured rates."""
    if a.no_tags:
        return None
    t = total_sec(parts)
    if stale_lang and rng.random() < a.reset_tag_stale:
        return stale_lang
    wrong = a.tag_wrong_onset if (t is not None and t <= a.onset_sec) else a.tag_wrong_steady
    if rng.random() < wrong:
        return rng.choice([x for x in LANGS if x != true_lang])
    return true_lang


def put_tag(row, tag):
    if tag is None:
        return row
    ins = f"speech {NAME[tag]} "
    if row.get("cond"):
        row["cond"] = ins + row["cond"]
    elif row.get("text", "").startswith("language "):
        row["text"] = ins + row["text"]
    else:
        row["cond"] = ins + (row.get("cond") or "")
    return row


out = open(a.out, "w", encoding="utf-8")
stats = {"carried": 0, "delta": 0, "ctx_attached": 0, "ctx_empty": 0, "ctx_dropout": 0,
         "reset_seg": 0, "reset_lid": 0, "transcribe": 0, "tag_wrong": 0}

delta_rows = []
for l in open(a.train_base, encoding="utf-8"):
    r = json.loads(l)
    parts = parse_audio(r["audio"])
    src_lang = lang_by_path.get(parts[0][0]) or r.get("pair", "x-x").split("-")[0]
    if not r.get("cond"):
        tag = tag_of(src_lang, parts)
        if tag and tag != src_lang:
            stats["tag_wrong"] += 1
        r.setdefault("prompt", "")
        out.write(json.dumps(put_tag(r, tag), ensure_ascii=False) + "\n")
        stats["carried"] += 1
        continue
    stats["delta"] += 1
    prompt = ""
    if not a.no_context and all(p in words_by_path or "silence" in p for p, _, _ in parts):
        if rng.random() < a.context_dropout:
            stats["ctx_dropout"] += 1
        else:
            lag = 2.5 if rng.random() < a.stale_frac else a.lag_sec
            ft = frontier_text(parts, lag)
            if ft:
                prompt = ft
                stats["ctx_attached"] += 1
            else:
                stats["ctx_empty"] += 1
    r["prompt"] = prompt
    tag = tag_of(src_lang, parts)
    if tag and tag != src_lang:
        stats["tag_wrong"] += 1
    r = put_tag(r, tag)
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    delta_rows.append((r, src_lang))

if not a.no_reset:
    pool = {}
    for p, lg in lang_by_path.items():
        if p in words_by_path and dur_by_path.get(p, 0) > 2.5:
            pool.setdefault(lg, []).append(p)
    cands = [(r, sl) for r, sl in delta_rows if "|" not in r["audio"]
             and r["cond"].endswith("<asr_text>")
             and os.path.abspath(r["audio"].split("#")[0]) in words_by_path]
    rng.shuffle(cands)
    for r, src_lang in cands[: a.reset_rows]:
        own = os.path.abspath(r["audio"].split("#")[0])
        lid = rng.random() < a.lid_frac
        tail_lang = rng.choice([lg for lg in pool if lg != src_lang]) if lid else src_lang
        tail = rng.choice(pool[tail_lang])
        if tail == own:
            continue
        keep = 1.5 if lid else 1.0
        td = dur_by_path[tail]
        audio = f"{tail}#start={max(0.0, td - keep):.2f}#end={td:.2f}|{r['audio']}"
        parts = parse_audio(audio)
        prompt = "" if a.no_context else (frontier_text(parts, a.lag_sec) or "")
        row = {k: v for k, v in r.items()}
        row["cond"] = row["cond"].split("language ", 1)[-1]
        row["cond"] = "language " + row["cond"]
        row.update(audio=audio, prompt=prompt)
        tag = tag_of(src_lang, parts, stale_lang=tail_lang if lid else None)
        if tag and tag != src_lang:
            stats["tag_wrong"] += 1
        out.write(json.dumps(put_tag(row, tag), ensure_ascii=False) + "\n")
        stats["reset_lid" if lid else "reset_seg"] += 1

if not a.no_transcribe:
    target = int(stats["delta"] * a.transcribe_ratio)
    per_lang = target // 2
    for lg in ("ko", "en"):
        paths = [p for p, l2 in lang_by_path.items()
                 if l2 == lg and p in words_by_path and p in src_by_path
                 and len(words_by_path[p]) >= 6]
        rng.shuffle(paths)
        made = 0
        for p in paths:
            if made >= per_lang:
                break
            ws = words_by_path[p]
            src = src_by_path[p]

            def _ok(i):
                if i >= len(ws):
                    return True
                b = _span_bounds(p, 0, i)
                return b is not None and b[1] + 1 < len(src) and not src[b[1] + 1].isalnum()

            rows_here = []
            full_cond = f"language {NAME[lg]}<asr_text>"
            rows_here.append({"audio": p, "cond": full_cond, "text": src, "pair": f"{lg}-{lg}"})
            for _ in range(a.rows_per_rec - 1):
                for _try in range(6):
                    j = rng.randint(0, max(0, len(ws) - 4))
                    k = rng.randint(j + 2, min(len(ws), j + 8))
                    if _ok(j) and _ok(k):
                        break
                else:
                    continue
                head = _src_span(p, 0, j) if j else ""
                cont = _src_span(p, j, k)
                if cont is None or (j and not head):
                    continue
                cut = ws[k - 1]["end"] + 0.2
                audio = p if k == len(ws) else f"{p}#end={cut:.3f}"
                rows_here.append({"audio": audio, "cond": full_cond + head,
                                  "text": (" " if j else "") + cont, "pair": f"{lg}-{lg}"})
            for row in rows_here:
                parts = parse_audio(row["audio"])
                row["prompt"] = ""
                tag = tag_of(lg, parts)
                if tag and tag != lg:
                    stats["tag_wrong"] += 1
                out.write(json.dumps(put_tag(row, tag), ensure_ascii=False) + "\n")
                made += 1
                stats["transcribe"] += 1

out.close()
tot = stats["carried"] + stats["delta"] + stats["reset_seg"] + stats["reset_lid"] + stats["transcribe"]
print(json.dumps(stats, indent=1))
print(f"{a.out}: {tot} rows, tag_wrong {stats['tag_wrong'] / max(1, tot):.1%}")
