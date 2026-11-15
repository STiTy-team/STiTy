"""Per-step emit/wait terms and a per-utterance latency floor from prefix14 emission schedules, for grpo_prefix.py.

Seed LiveInterpret 2.0 (arXiv 2507.17527) scores every chunk against a reference chunk: a reward for staying quiet
when the reference is empty, a reward for speaking when the reference speaks, and a lagging term floored at a
wait threshold so latency cannot be bought below what the reference itself needs. The prefix14 schedule
[(after_word, text), ...] plus word end times is exactly such a reference on the rollout grid
(t = first, first + step, ..., dur):

  due steps      a step whose window (t_prev, t] contains an item's emit time (end of word + pad). Items due
                 before the first step land on the first step; items due after the audio ends land on the last.
  wait steps     a step with nothing due in its window and nothing due in the previous window (a one-step lag
                 is forgiven); every step past the real audio end is a wait step (silence tail).
  emit_agree     share of due steps where the rollout emitted text              (Seed r^s)
  wait_agree     share of wait steps where the rollout emitted nothing          (Seed r^l)
  lat_label      the schedule's own latency on the same scale as the rollout's: mean over label characters of
                 emit time / dur. grpo_prefix.py charges lam * max(lat, lat_label) so no reward lies below it.

Pure python: no torch, no audio."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "labels"))
from build_prefix14 import emit_times, end_times_for, split_holdout  # noqa: E402
from schedule_gate import _SENT_CLOSE, is_bare_subject, words  # noqa: E402

EPS = 1e-9


def parse_clip(audio):
    """'path#start=S#end=E' -> (path, start or None, end or None)."""
    path, _, params = audio.partition("#")
    kv = dict(p.partition("=")[::2] for p in params.split("#")) if params else {}
    return path, (float(kv["start"]) if "start" in kv else None), (float(kv["end"]) if "end" in kv else None)


def schedule_grid(dur, first, step):
    """Rollout decode times: first, first + step, ... capped at dur, then dur itself (grpo_prefix.rollout)."""
    pts, t = [], min(first, dur)
    while True:
        pts.append(round(t, 3))
        if t >= dur - EPS:
            break
        t = min(dur, t + step)
    return pts


def step_labels(due, dur_real, grid):
    """For each grid time: 'due' if an item becomes due in its window, 'wait' if nothing is due there or in the
    previous window (or the step is past the real audio), else None (lagging step, unscored). Items due after the
    real audio ends (end of last word + pad) are due at the first step that has heard all of it, as rows_end trains."""
    out, prev_t, prev_due = [], 0.0, False
    due = [min(e, dur_real) for e in due]
    for i, t in enumerate(grid):
        is_due = any((prev_t < e <= t + EPS) or (i == 0 and e <= t + EPS) for e in due)
        if t > dur_real + EPS and not is_due:
            out.append("wait")
        elif is_due:
            out.append("due")
        elif prev_due:
            out.append(None)
        else:
            out.append("wait")
        prev_t, prev_due = t, is_due
    return out


def agreement(labels, emitted):
    """(emit_agree, wait_agree) for parallel lists of step labels and emitted flags. A class with no steps scores 1."""
    due = [e for l, e in zip(labels, emitted) if l == "due"]
    wait = [e for l, e in zip(labels, emitted) if l == "wait"]
    s = sum(due) / len(due) if due else 1.0
    w = sum(1 for e in wait if not e) / len(wait) if wait else 1.0
    return s, w


def label_latency(due, chars, dur):
    """Mean over label characters of emit time / dur: the schedule's own lag on the rollout's scale."""
    n = sum(chars)
    return sum(min(e, dur) * c for e, c in zip(due, chars)) / n / dur if n and dur > 0 else 0.0


def latency_charge(lat, lat_label, scale=1.0):
    """The latency the reward charges: the rollout's own, floored at `scale` x the schedule's. scale 1 = never
    below the labels (prefix14/15); scale < 1 lets a rollout earn by emitting earlier than labels that wait."""
    return max(lat, lat_label * scale)


def shape_terms(pieces, src_close, tgt):
    """(early_close, bare_first) for one rollout. pieces = [(t, text)] of its non-empty additions in order;
    src_close = end times of the source words that close a sentence (the last word always does).
    early_close: share of closing pieces whose close outruns the source - the n-th target close emitted before the
    n-th source close was heard (prefix15's failure: the label rule 'close only when the source closed').
    bare_first: 1 when the first of several pieces is a bare subject ("그는", "They") - label rule 4."""
    closes = [(t, x) for t, x in pieces if _SENT_CLOSE.search(x.strip())]
    early = 0
    for n, (t, _) in enumerate(closes):
        heard = sum(1 for c in src_close if c <= t + EPS)
        early += n + 1 > heard
    bare = float(len(pieces) > 1 and is_bare_subject(pieces[0][1].strip(), tgt, first=True))
    return (early / len(closes) if closes else 0.0), bare


def load_pool(schedule, sentences, word_times, pairs, max_dur, holdout=0.02, seed=14):
    """Utterances of the schedule set (train split only) as grpo_prefix pool dicts plus 'due' (emit times of the
    non-empty items), 'chars' (their character counts without spaces) and 'src_close' (end times of the source
    words that close a sentence). Skipped: rescued wait labels (prefix16 'rescue': 'wait' - the whole sentence at
    the end, the opposite of what GRPO is asked to learn) and sentences without a reference (COVoST en-ko)."""
    sched = {}
    for l in open(schedule, encoding="utf-8"):
        r = json.loads(l)
        if r.get("rescue") == "wait":
            continue
        sched[r["sid"]] = [(int(x), t) for x, t in r["schedule"]]
    wt = {json.loads(l)["audio"]: json.loads(l)["words"] for l in open(word_times, encoding="utf-8") if l.strip()}
    sents = [json.loads(l) for l in open(sentences, encoding="utf-8") if l.strip()]
    sents = [s for s in sents if s["sid"] in sched and any(t.strip() for _, t in sched[s["sid"]]) and s.get("ref")]
    held = split_holdout([s["sid"] for s in sents], holdout, seed) if holdout > 0 else set()
    pool = []
    for s in sents:
        pair = f"{s['lang']}-{s['tgt']}"
        if pair not in pairs or s["sid"] in held:
            continue
        toks = words(s["src"], s["lang"])
        items = [(a, t) for a, t in sched[s["sid"]] if t.strip()]
        for rec in s["recs"]:
            ws = wt.get(rec["audio"])
            if not ws:
                continue
            _, start, end = parse_clip(rec["audio"])
            if start is not None and end is not None and end - start > max_dur:
                continue
            ends = end_times_for(toks, ws)
            close = [e for i, (w, e) in enumerate(zip(toks, ends)) if _SENT_CLOSE.search(w) or i == len(toks) - 1]
            pool.append({"pair": pair, "tgt": s["tgt"], "wav": rec["audio"], "src": s["src"], "ref": s["ref"],
                         "due": emit_times(items, ends), "chars": [len("".join(t.split())) for _, t in items],
                         "src_close": close})
    random.Random(seed).shuffle(pool)
    return pool
