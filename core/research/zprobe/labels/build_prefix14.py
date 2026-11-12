"""Build the prefix14 SFT corpus from emission schedules (one row = audio cut at t, text emitted so far, new text).

A schedule is [(after_word, text), ...]: once source word `after_word` has been heard the text is
appended to the translation. Word end times (forced alignment, per recording) turn that into rows:

  on_time     audio cut on a 0.5s grid plus each item's emit time (end of word + pad), before dur;
              cond = items due at the previous cut, text = items that became due since. Mostly empty.
  catchup     cond is pulled back 1-2 items behind what the audio supports; text = the backlog.
  end         (a) cut exactly at the end of speech, cond = state one cut earlier, text = everything left;
              (b) the whole sentence: cond = prefix only, text = all items joined (uncut audio of the
              sentence's first recording, once per sentence).
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
from schedule_gate import eojeol_end_times, glues, join_texts, words  # noqa: E402

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
    return join_texts([t for _, t in items])


def _delta(cond_items, new_items):
    """Text appended to the cond text: a leading space unless cond is empty or the first new piece glues on
    (a particle/ending continuing the previous eojeol: 응답률 + 도)."""
    d = _join(new_items)
    c = _join(cond_items)
    if not d or not c:
        return d
    first = next((t for _, t in new_items if t.strip()), "")
    return d if glues(c, first) else " " + d


PAD = 0.2


def heard(end_times, t, pad=PAD):
    """Words heard at t: a word counts only after its end + pad (alignment error margin)."""
    return sum(1 for e in end_times if e + pad <= t + EPS)


def emit_times(items, end_times, pad=PAD):
    if any(a < 1 for a, _ in items):
        raise ValueError(f"schedule item with after < 1: {items}")
    return [end_times[min(a, len(end_times)) - 1] + pad for a, _ in items]


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


def _cap_keep(rows, cap, rng):
    """Indices to keep under a per-recording cap. ceil(cap/3) slots are reserved for empty-text (wait)
    rows, preferring the grid point just before each emission (the hardest "not yet" decision); the
    rest go to emission rows; unused slots on either side go to the other."""
    is_em = [r["_emit"] or bool(r["text"]) for r in rows]
    em = [i for i, e in enumerate(is_em) if e]
    pre = sorted({i - 1 for i in em if i > 0 and not is_em[i - 1]})
    other = [i for i, e in enumerate(is_em) if not e and i not in set(pre)]
    rng.shuffle(em)
    rng.shuffle(pre)
    rng.shuffle(other)
    empty = pre + other
    n_empty = min(len(empty), -(-cap // 3))
    n_em = min(len(em), cap - n_empty)
    n_empty = min(len(empty), cap - n_em)
    return sorted(em[:n_em] + empty[:n_empty])


def rows_on_time(items, end_times, dur, pair, audio, cap=None, rng=None):
    tag = tag_of(pair)
    emits = {round(e, 3) for (a, t), e in zip(items, emit_times(items, end_times)) if t.strip()}
    rows, prev_w = [], 0
    for t in grid(dur, emit_times(items, end_times)):
        if t >= dur - EPS:
            break  # the cut at dur belongs to rows_end (its cond lags one emission)
        w = heard(end_times, t)
        cond_items = [(a, x) for a, x in items if a <= prev_w]
        new_items = [(a, x) for a, x in items if prev_w < a <= w]
        r = _row(cut(audio, t), tag + _join(cond_items), _delta(cond_items, new_items), pair)
        r["_emit"] = t in emits
        rows.append(r)
        prev_w = w
    if cap is not None and len(rows) > cap:
        rows = [rows[i] for i in _cap_keep(rows, cap, rng or random.Random(0))]
    for r in rows:
        r.pop("_emit")
    return rows


def rows_catchup(items, end_times, dur, pair, audio, rng, n=2, lag=(1, 2)):
    """Cut on the plain 0.5s grid. cond = the on-time cond at the previous cut minus its last L items;
    text = those L items + items newly due at t."""
    tag = tag_of(pair)
    full = grid(dur, emit_times(items, end_times))
    cand = []
    for t in grid(dur, []):
        k = full.index(t)
        prev_w = heard(end_times, full[k - 1]) if k else 0
        cond_items = _due(items, prev_w)
        ls = [x for x in lag if x <= len(cond_items)]
        if ls:
            cand.append((t, prev_w, cond_items, ls))
    rng.shuffle(cand)
    out = []
    for t, prev_w, cond_items, ls in cand[:n]:
        L = rng.choice(ls)
        w = heard(end_times, t)
        back = cond_items[-L:] + [(a, x) for a, x in items if prev_w < a <= w and x.strip()]
        keep = cond_items[:-L]
        out.append(_row(cut(audio, t), tag + _join(keep), _delta(keep, back), pair))
    return out


def rows_end(items, pair, audio, dur, end_times, whole=True):
    """(a) cut at the end of speech: cond = on-time state just before the last pending emission (at least
    one item short), text = the rest. (b) whole sentence: cond = prefix only, text = all items; only when
    `whole` (main passes it for the first recording of a sentence, so the row is not repeated per recording)."""
    tag = tag_of(pair)
    its = [(a, t) for a, t in items if t.strip()]
    ems = emit_times(its, end_times)
    if its:
        last_e = ems[-1]
        keep = [it for it, e in zip(its[:-1], ems[:-1]) if e < last_e - EPS]
    else:
        keep = []
    rest = its[len(keep):]
    out = [_row(cut(audio, dur), tag + _join(keep), _delta(keep, rest), pair)]
    if whole:
        out.append(_row(audio, tag, _join(items), pair))
    return out


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
    """recs_a: [{"sid", "audio", "full"}]; recs_b: [{"sid", "audio", "first_end"}] (same length, paired by index)."""
    tag = tag_of(pair)
    out = []
    for A, B in zip(recs_a, recs_b):
        if A.get("sid") is not None and A.get("sid") == B.get("sid"):
            continue
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


def audio_paths(audio):
    """File paths inside an audio string: `|`-joined pieces, each `path[#start=..][#end=..]`."""
    return [p.partition("#")[0] for p in audio.split("|")]


def drop_missing(rows, exists=os.path.exists):
    """(rows whose every audio piece exists, sorted missing paths)."""
    cache, missing, keep = {}, set(), []
    for r in rows:
        ok = True
        for p in audio_paths(r["audio"]):
            if p not in cache:
                cache[p] = exists(p)
            if not cache[p]:
                ok = False
                missing.add(p)
        if ok:
            keep.append(r)
    return keep, sorted(missing)


def write_jsonl_atomic(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


ROW_RANGE = (60000, 200000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="models/zprobe/data/prefix14/schedule_luna.gated.jsonl")
    ap.add_argument("--sentences", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--word-times", default="models/zprobe/data/prefix14/word_times.jsonl")
    ap.add_argument("--sil-wav", default="models/zprobe/data/silence_tail_4s.wav")
    ap.add_argument("--out", default="models/zprobe/data/train_prefix14.jsonl")
    ap.add_argument("--val-out", default="models/zprobe/data/val_prefix14.jsonl")
    ap.add_argument("--rows-per-rec", type=int, default=5)
    ap.add_argument("--catchup-per-rec", type=int, default=2)
    ap.add_argument("--sil-cap", type=int, default=800, help="per pair (en-ko gets twice this)")
    ap.add_argument("--seg-cap", type=int, default=2500, help="per pair (en-ko gets twice this)")
    ap.add_argument("--reset-rows", type=int, default=3000)
    ap.add_argument("--lid-frac", type=float, default=0.33)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--val-cap", type=int, default=1500, help="random sample of held-out rows (0 = all)")
    ap.add_argument("--no-range-check", action="store_true",
                    help=f"skip the total row-count check {ROW_RANGE[0]}..{ROW_RANGE[1]}")
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sil_wav = os.path.abspath(a.sil_wav)

    sched = {}
    for l in open(a.schedule, encoding="utf-8"):
        r = json.loads(l)
        sched[r["sid"]] = [(int(x), t) for x, t in r["schedule"]]
        if any(x < 1 for x, _ in sched[r["sid"]]):
            raise ValueError(f"sid {r['sid']}: schedule item with after < 1")
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

    dur_cache, skipped, gone = {}, 0, []
    recs = []  # one dict per usable recording
    for s in sents:
        toks = words(s["src"], s["lang"])
        for rec in s["recs"]:
            ws = wt.get(rec["audio"])
            if not ws:
                skipped += 1
                continue
            if not os.path.exists(audio_paths(rec["audio"])[0]):
                gone.append(rec["audio"])
                continue
            recs.append(dict(sid=s["sid"], lang=s["lang"], pair=f"{s['lang']}-{s['tgt']}",
                             audio=rec["audio"], items=sched[s["sid"]],
                             ends=end_times_for(toks, ws), dur=_clip_dur(rec["audio"], dur_cache),
                             held=s["sid"] in held))

    train, val, fam_of = [], [], {}  # fam_of: id(row) -> family name, for the report

    def add(dst, fam, rows):
        for r in rows:
            dst.append(r)
            fam_of[id(r)] = fam

    on_time_train, whole_done = [], set()
    for r in recs:
        dst = val if r["held"] else train
        ot = rows_on_time(r["items"], r["ends"], r["dur"], r["pair"], r["audio"], a.rows_per_rec, rng)
        add(dst, "on_time", ot)
        if not r["held"]:
            on_time_train += ot
        add(dst, "catchup", rows_catchup(r["items"], r["ends"], r["dur"], r["pair"], r["audio"], rng,
                                         a.catchup_per_rec))
        add(dst, "end", rows_end(r["items"], r["pair"], r["audio"], r["dur"], r["ends"], whole=r["sid"] not in whole_done))
        whole_done.add(r["sid"])

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
            [{"sid": r["sid"], "audio": r["audio"], "full": _join(r["items"])} for r in A[:m]],
            [{"sid": r["sid"], "audio": r["audio"], "first_end": r["ends"][0]} for r in B[:m]], pair, a.seg_cap * mult))

    rpool = {}
    for r in recs:
        if not r["held"] and "#" not in r["audio"] and "|" not in r["audio"] and r["dur"] > 2.5:
            rpool.setdefault(r["lang"], []).append((r["audio"], r["dur"]))
    rr, fam = rows_reset(on_time_train, rpool, rng, a.reset_rows, a.lid_frac)
    for row, f in zip(rr, fam):
        add(train, f, [row])

    rng.shuffle(train)
    rng.shuffle(val)
    if a.val_cap and len(val) > a.val_cap:
        print(f"val: sampled {a.val_cap} of {len(val)} held-out rows")
        val = val[: a.val_cap]  # already shuffled with the seeded rng

    # one unreadable file would kill lazy-feature SFT hours in: check every piece of every row now
    n_all = len(train) + len(val)
    train, miss_t = drop_missing(train)
    val, miss_v = drop_missing(val)
    missing = sorted(set(miss_t) | set(miss_v))
    dropped = n_all - len(train) - len(val)
    n_recs = len(recs) + len(gone)
    if gone:
        print(f"missing audio: {len(gone)}/{n_recs} recordings skipped; e.g. {gone[:5]}")
    if missing:
        print(f"missing audio: {len(missing)} files, {dropped} rows dropped; e.g. {missing[:5]}")
    if dropped > 0.005 * max(1, n_all) or len(gone) > 0.005 * max(1, n_recs):
        print(f"BUILD_PREFIX14_FAIL missing audio: {len(gone)}/{n_recs} recordings, {dropped}/{n_all} rows (> 0.5%)")
        sys.exit(1)

    for name, rows in (("train", train), ("val", val)):
        fams = {}
        for r in rows:
            fams[fam_of[id(r)]] = fams.get(fam_of[id(r)], 0) + 1
        fams = dict(sorted(fams.items()))
        empty = sum(1 for r in rows if not r["text"])
        print(f"{name}: {len(rows)} rows; per family {fams}; empty text {empty} ({empty / max(1, len(rows)):.1%})")
    held_recs = sum(1 for r in recs if r["held"])
    print(f"holdout: {len(held)} sentences, {held_recs} recs; recs skipped (no word_times): {skipped}")
    total = len(train) + len(val)
    print(f"total: train {len(train)} + val {len(val)} = {total}")
    lo, hi = ROW_RANGE
    if not a.no_range_check and not lo <= total <= hi:
        print(f"BUILD_PREFIX14_FAIL total rows {total} outside [{lo}, {hi}] - change --rows-per-rec "
              "or pass --no-range-check")
        sys.exit(1)
    write_jsonl_atomic(a.val_out, val)
    write_jsonl_atomic(a.out, train)  # last: the chain treats the train file as the done marker
    print("BUILD_PREFIX14_DONE")


if __name__ == "__main__":
    main()
