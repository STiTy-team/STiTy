"""Build the prefix14 SFT corpus from emission schedules (one row = audio cut at t, text emitted so far, new text).

A schedule is [(after_word, text), ...]: once source word `after_word` has been heard the text is
appended to the translation. Word end times (forced alignment, per recording) turn that into rows:

  on_time     audio cut on a 0.5s grid plus each item's emit time (end of word + pad);
              cond = items due at the previous cut, text = items that became due since. Mostly empty.
  catchup     cond is pulled back 1-2 items behind what the audio supports; text = the backlog.
  end         (a) cut exactly at the end of speech, cond = state one cut earlier, text = everything left;
              (b) the whole sentence: cond = prefix only, text = all items joined (uncut audio).
  sil_end     recording | quiet tail, cond = everything, text = "" (no hallucination in silence).
  sil_catchup recording | quiet tail, cond = all but the last L items (L in 1..n), text = those L.
  seg         recording A | start of recording B, cond = A's full text, text = " <SEG>".
  reset_seg / reset_lid
              tail of another recording (same / other language) glued in front of an on_time row
              that has nothing emitted yet; same cond/text, no tag, no prompt.

Held-out sentences (all recs of a sid go to the same side) get on_time/catchup/end rows only and form
the streaming validation set.

usage: .venv/bin/python core/research/zprobe/labels/build_prefix14.py [--schedule ... --out ...]"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import eojeol_end_times, words  # noqa: E402

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
EPS = 1e-9


def end_times_for(tokens, word_times_words):
    return eojeol_end_times(tokens, word_times_words)


def tag_of(pair):
    return f"language {NAME[pair.split('-')[1]]}<asr_text>"


def cut(audio, t):
    """Cut a clip at t seconds from its start. Clips with #start=S#end=E stay inside [S, E]."""
    path, _, params = audio.partition("#")
    start, end = None, None
    for p in params.split("#") if params else []:
        k, _, v = p.partition("=")
        if k == "start":
            start = float(v)
        elif k == "end":
            end = float(v)
    s = start or 0.0
    new_end = s + t
    if end is not None:
        new_end = min(new_end, end)
    if start is not None:
        return f"{path}#start={start:.3f}#end={new_end:.3f}"
    return f"{path}#end={new_end:.3f}"


def _join(items):
    return " ".join(t.strip() for _, t in items if t.strip())


def _delta(cond_items, new_items):
    d = _join(new_items)
    return (" " + d) if d and _join(cond_items) else d


def heard(end_times, t):
    return sum(1 for e in end_times if e <= t + EPS)


def emit_times(items, end_times, pad=0.2):
    return [end_times[min(max(a, 1), len(end_times)) - 1] + pad for a, _ in items]


def grid(dur, emit_ts, step=0.5, first=1.0):
    pts = set()
    k = 1
    while k * step <= dur + EPS:
        if k * step >= first - EPS:
            pts.add(round(k * step, 3))
        k += 1
    for e in emit_ts:
        if first - EPS <= e <= dur + EPS:
            pts.add(round(e, 3))
    return sorted(pts)


def _row(audio, cond, text, pair):
    return {"audio": audio, "cond": cond, "text": text, "pair": pair, "prompt": ""}


def _due(items, w):
    return [(a, t) for a, t in items if t.strip() and a <= w]


def rows_on_time(items, end_times, dur, pair, audio, cap=None, rng=None):
    tag = tag_of(pair)
    emits = {round(e, 3) for (a, t), e in zip(items, emit_times(items, end_times)) if t.strip()}
    rows, prev_w = [], 0
    for t in grid(dur, emit_times(items, end_times), ):
        w = heard(end_times, t)
        cond_items = [(a, x) for a, x in items if a <= prev_w]
        new_items = [(a, x) for a, x in items if prev_w < a <= w]
        r = _row(cut(audio, t), tag + _join(cond_items), _delta(cond_items, new_items), pair)
        r["_emit"] = t in emits
        rows.append(r)
        prev_w = w
    if cap is not None and len(rows) > cap:
        rng = rng or random.Random(0)
        em = [i for i, r in enumerate(rows) if r["_emit"] or r["text"]]
        rest = [i for i, r in enumerate(rows) if not (r["_emit"] or r["text"])]
        rng.shuffle(em)
        rng.shuffle(rest)
        keep = sorted((em + rest)[:cap])
        rows = [rows[i] for i in keep]
    for r in rows:
        r.pop("_emit")
    return rows


def rows_catchup(items, end_times, dur, pair, audio, rng, n=2, lag=(1, 2)):
    tag = tag_of(pair)
    pts = grid(dur, emit_times(items, end_times))
    out, avail = [], list(pts)
    rng.shuffle(avail)
    for t in avail:
        if len(out) >= n:
            break
        due = _due(items, heard(end_times, t))
        ls = [x for x in lag if x <= len(due)]
        if not ls:
            continue
        L = rng.choice(ls)
        cond_items, back = due[:-L], due[-L:]
        out.append(_row(cut(audio, t), tag + _join(cond_items), _delta(cond_items, back), pair))
    return out


def rows_end(items, pair, audio, dur, end_times):
    tag = tag_of(pair)
    pts = grid(dur, emit_times(items, end_times))
    prev_w = heard(end_times, pts[-2]) if len(pts) >= 2 else 0
    cond_items = [(a, x) for a, x in items if a <= prev_w]
    rest = [(a, x) for a, x in items if a > prev_w]
    return [_row(cut(audio, dur), tag + _join(cond_items), _delta(cond_items, rest), pair),
            _row(audio, tag, _join(items), pair)]


def rows_silence(items, pair, audio, sil_wav, rng, dur, catchup=False):
    tag = tag_of(pair)
    a = f"{cut(audio, dur)}|{sil_wav}#end={rng.uniform(0.8, 2.0):.2f}"
    its = [(i, t) for i, t in items if t.strip()]
    if not catchup or len(its) < 1:
        return _row(a, tag + _join(its), "", pair)
    L = rng.randint(1, len(its))
    cond_items, back = its[:-L], its[-L:]
    return _row(a, tag + _join(cond_items), _delta(cond_items, back), pair)


def rows_seg_pairs(recs_a, recs_b, pair, cap=None):
    """recs_a: [{"audio", "full"}]; recs_b: [{"audio", "first_end"}] (same length, paired by index)."""
    tag = tag_of(pair)
    out = []
    for A, B in zip(recs_a, recs_b):
        if cap is not None and len(out) >= cap:
            break
        t_b = max(0.3, B["first_end"] * 0.5)
        out.append(_row(f"{A['audio']}|{cut(B['audio'], t_b)}", tag + A["full"], " <SEG>", pair))
    return out


def rows_reset(rows, pool, rng, n, lid_frac=0.33):
    """pool: {lang: [(plain_path, duration)]}. Glue the tail of another recording in front of on_time rows."""
    cands = [r for r in rows if "|" not in r["audio"] and r["cond"].endswith("<asr_text>")]
    rng.shuffle(cands)
    out, fam = [], []
    for r in cands:
        if len(out) >= n:
            break
        src_lang = r["pair"].split("-")[0]
        others = [lg for lg in pool if lg != src_lang and pool[lg]]
        lid = rng.random() < lid_frac and bool(others)
        tl = rng.choice(others) if lid else src_lang
        if not pool.get(tl):
            continue
        tail, td = rng.choice(pool[tl])
        if tail == r["audio"].split("#")[0]:
            continue
        keep = 1.5 if lid else 1.0
        audio = f"{tail}#start={max(0.0, td - keep):.2f}#end={td:.2f}|{r['audio']}"
        out.append(dict(r, audio=audio))
        fam.append("reset_lid" if lid else "reset_seg")
    return out, fam


def split_holdout(sids, frac=0.02, seed=14):
    u = sorted(set(sids))
    k = max(1, round(len(u) * frac)) if u else 0
    return set(random.Random(seed).sample(u, k))


def _clip_dur(audio, cache):
    path, _, params = audio.partition("#")
    kv = dict(p.partition("=")[::2] for p in params.split("#")) if params else {}
    if "start" in kv and "end" in kv:
        return float(kv["end"]) - float(kv["start"])
    if path not in cache:
        import soundfile as sf
        cache[path] = sf.info(path).duration
    d = cache[path]
    return d - float(kv.get("start", 0)) if "start" in kv else (min(d, float(kv["end"])) if "end" in kv else d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="models/zprobe/data/prefix14/schedule_luna.gated.jsonl")
    ap.add_argument("--sentences", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--word-times", default="models/zprobe/data/prefix14/word_times.jsonl")
    ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
    ap.add_argument("--out", default="models/zprobe/data/train_prefix14.jsonl")
    ap.add_argument("--val-out", default="models/zprobe/data/val_prefix14.jsonl")
    ap.add_argument("--rows-per-rec", type=int, default=8)
    ap.add_argument("--catchup-per-rec", type=int, default=2)
    ap.add_argument("--sil-cap", type=int, default=800, help="per pair (en-ko gets twice this)")
    ap.add_argument("--seg-cap", type=int, default=2500, help="per pair (en-ko gets twice this)")
    ap.add_argument("--reset-rows", type=int, default=3000)
    ap.add_argument("--lid-frac", type=float, default=0.33)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sil_wav = os.path.abspath(a.sil_wav)

    sched = {}
    for l in open(a.schedule, encoding="utf-8"):
        r = json.loads(l)
        sched[r["sid"]] = [(int(x), t) for x, t in r["schedule"]]
    wt = {}
    for l in open(a.word_times, encoding="utf-8"):
        r = json.loads(l)
        wt[r["audio"]] = r["words"]
    sents = []
    for l in open(a.sentences, encoding="utf-8"):
        r = json.loads(l)
        if r["sid"] in sched and any(t.strip() for _, t in sched[r["sid"]]):
            sents.append(r)
    held = split_holdout([s["sid"] for s in sents], a.holdout, a.seed)

    dur_cache, skipped = {}, 0
    recs = []  # one dict per usable recording
    for s in sents:
        toks = words(s["src"], s["lang"])
        for rec in s["recs"]:
            ws = wt.get(rec["audio"])
            if not ws:
                skipped += 1
                continue
            recs.append(dict(sid=s["sid"], lang=s["lang"], pair=f"{s['lang']}-{s['tgt']}",
                             audio=rec["audio"], items=sched[s["sid"]],
                             ends=end_times_for(toks, ws), dur=_clip_dur(rec["audio"], dur_cache),
                             held=s["sid"] in held))

    train, val, fam_n = [], [], {}

    def add(dst, fam, rows):
        for r in rows:
            dst.append(r)
            fam_n[(dst is val, fam)] = fam_n.get((dst is val, fam), 0) + 1

    on_time_train = []
    for r in recs:
        dst = val if r["held"] else train
        ot = rows_on_time(r["items"], r["ends"], r["dur"], r["pair"], r["audio"], a.rows_per_rec, rng)
        add(dst, "on_time", ot)
        if not r["held"]:
            on_time_train += ot
        add(dst, "catchup", rows_catchup(r["items"], r["ends"], r["dur"], r["pair"], r["audio"], rng,
                                         a.catchup_per_rec))
        add(dst, "end", rows_end(r["items"], r["pair"], r["audio"], r["dur"], r["ends"]))

    by_pair = {}
    for r in recs:
        if not r["held"]:
            by_pair.setdefault(r["pair"], []).append(r)
    for pair, rs in sorted(by_pair.items()):
        mult = 2 if pair == "en-ko" else 1
        pool = rng.sample(rs, min(a.sil_cap * mult, len(rs)))
        for n, r in enumerate(pool):
            c = n % 2 == 1
            row = rows_silence(r["items"], pair, r["audio"], sil_wav, rng, r["dur"], catchup=c)
            add(train, "sil_catchup" if c and row["text"] else "sil_end", [row])
        sp = [r for r in rs if _join(r["items"])]
        rng.shuffle(sp)
        A, B = sp[0::2], sp[1::2]
        m = min(len(A), len(B))
        add(train, "seg", rows_seg_pairs(
            [{"audio": r["audio"], "full": _join(r["items"])} for r in A[:m]],
            [{"audio": r["audio"], "first_end": r["ends"][0]} for r in B[:m]], pair, a.seg_cap * mult))

    rpool = {}
    for r in recs:
        if not r["held"] and "#" not in r["audio"] and "|" not in r["audio"] and r["dur"] > 2.5:
            rpool.setdefault(r["lang"], []).append((r["audio"], r["dur"]))
    rr, fam = rows_reset(on_time_train, rpool, rng, a.reset_rows, a.lid_frac)
    for row, f in zip(rr, fam):
        add(train, f, [row])

    rng.shuffle(train)
    rng.shuffle(val)
    for path, rows in ((a.out, train), (a.val_out, val)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for isval, name, rows in ((False, "train", train), (True, "val", val)):
        fams = {f: n for (v, f), n in sorted(fam_n.items()) if v == isval}
        empty = sum(1 for r in rows if not r["text"])
        print(f"{name}: {len(rows)} rows {fams}; empty text {empty} ({empty / max(1, len(rows)):.1%})")
    held_recs = sum(1 for r in recs if r["held"])
    print(f"holdout: {len(held)} sentences, {held_recs} recs; recs skipped (no word_times): {skipped}")
    print("BUILD_PREFIX14_DONE")


if __name__ == "__main__":
    main()
