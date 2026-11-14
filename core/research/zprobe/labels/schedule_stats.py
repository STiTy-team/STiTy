"""Completeness gate + label hygiene table for emission-schedule labels.

1. Gate: keep a schedule only if its joined text is a complete translation.
   - ref present: COMET-da(src, joined, ref) >= --comet_min (0.70)
   - ref null:    cometkiwi(joined) >= cometkiwi(Hy full translation) - --kiwi_margin (0.08)
   Survivors (with "comet" or "kiwi"/"kiwi_hy" added) go to --out-gated. --skip-completeness copies every row
   (CPU tests / stats-only check; comet is imported lazily so it is not needed then).
2. Hygiene table per direction lang->tgt (empty-text items are waits and are ignored; a direction the gate
   dropped entirely gets a row with n=0): n, empty_share, dump_share, first_after_median/p95, deferral_median/max,
   len_ratio_median/p5/p95, gated_out. With --old, the same metrics for the old phrase-level corpus (aligned.jsonl).
   Old-corpus approximations: phrases are not word-delimited, so the source tokens are split proportionally to
   the phrase index (phrase i of P owns tokens round(i*N/P)..round((i+1)*N/P)); "words to first item" = tokens
   before the first non-empty addition; deferral is not available ("-"); empty_share = empty adds / all adds;
   dump_share = recordings with >= 2 phrases whose every add but the last is empty.

usage (gate needs comet, so use the tpm env; GPU):
  .venv-tpm/bin/python core/research/zprobe/labels/schedule_stats.py --schedule schedule_luna.jsonl \
    --sentences models/zprobe/data/prefix14/sentences.jsonl --hy models/zprobe/data/prefix14/hy_prefixes.jsonl \
    --old models/zprobe/data/prefix13/aligned.jsonl --out-gated schedule_luna.gated.jsonl
  add --skip-completeness for a CPU-only stats run.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re

from schedule_gate import _tags, clause_ends, join_texts, words  # noqa: E402

EN_CONNECTIVES = {"and", "or", "but", "so", "oh", "well", "then", "next", "still", "anyway", "also", "now"}
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_ABBREV_END = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|vs|etc|Inc|Ltd|Co|U\.S|e\.g|i\.e)\.$")
_PUNCT_TOKEN = re.compile(r"[.?!][\"'”’)\]]*$")


def is_connective_piece(text: str, tgt: str) -> bool:
    """A piece that is only a connective or interjection (그리고 / So, / Oh) - allowed by rule 3, so it is left out of
    the short-piece share."""
    t = text.strip().rstrip(".,!?…")
    if not t or " " in t:
        return False
    if tgt == "ko":
        tags = _tags(t)
        return bool(tags) and all(x in ("MAJ", "IC") for x in tags)
    return t.lower() in EN_CONNECTIVES


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def stats_for(rows, tokens_fn, known_fn=None, content_fn=None, gated_out=0, word_times=None, points_fn=None):
    """rows: dicts with sid, lang, tgt, src, schedule [[after, text], ...]. Returns the metric dict.
    prefix16 columns: mid_closed_share (non-final pieces ending a sentence), first_bare_subj (sentences whose first
    piece is a bare subject or, with src_words, covers < 3 words), first_after_frac (first emission / K, median),
    point_empty_share (empty items / all items - the labeler's waits), regret_share (regretted pieces / all candidate
    items - one piece per regret round), est_lag_s (mean over source words of: end of the word -> end of the `after`
    word of the piece that carries it; needs word_times {sid: words})."""
    from schedule_gate import _SENT_CLOSE, FIRST_PIECE_MIN_WORDS, eojeol_end_times, is_bare_subject
    empties, firsts, defs, ratios, lens, per_word = [], [], [], [], [], []
    dump = big = short = short_tot = single = 0
    mid_tot = mid_closed = bare = pt_tot = pt_empty = regret = 0
    first_frac, lags = [], []
    for r in rows:
        if r.get("rescue") == "wait":
            raise ValueError("stats_for: wait labels are counted separately, not in the corpus table")
        toks = tokens_fn(r["src"], r["lang"])
        K = len(toks)
        sch = [(int(a), t) for a, t in r["schedule"] if t.strip()]  # empty items are waits, not emissions
        pt_tot += len(r["schedule"])
        pt_empty += sum(1 for _, t in r["schedule"] if not t.strip())
        regret += r.get("regret_rounds") or 0
        for _, t in sch[:-1]:
            mid_tot += 1
            mid_closed += bool(_SENT_CLOSE.search(t.strip()))
        if sch:
            cov0 = next((c for (_, t), c in zip(r["schedule"], r.get("src_words") or [None] * len(r["schedule"])) if t.strip()), None)
            narrow = cov0 is not None and len(cov0) < FIRST_PIECE_MIN_WORDS and len(sch) > 1 and not sch[0][1].strip().endswith(",") \
                and not is_connective_piece(sch[0][1], r["tgt"])
            bare += is_bare_subject(sch[0][1], r["tgt"]) or narrow
            if K:
                first_frac.append(min(a for a, _ in sch) / K)
        if word_times is not None and r.get("src_words") and r["sid"] in word_times:
            ends = eojeol_end_times(toks, word_times[r["sid"]])
            for (a, t), c in zip(r["schedule"], r["src_words"]):
                if t.strip() and c and 1 <= a <= K:
                    lags += [ends[a - 1] - ends[j - 1] for j in c if 1 <= j <= K]
        if K and sch:
            empties.append(1 - len({a for a, _ in sch}) / K)
            firsts.append(min(a for a, _ in sch))
            per_word.append(len(sch) / K)
        single += len(sch) == 1
        for _, t in sch:
            lens.append(len(t.strip()))
            if not is_connective_piece(t, r["tgt"]):
                short_tot += 1
                short += len(t.strip()) <= 3
        if K >= 4:
            big += 1
            dump += bool(sch) and all(a > 2 * K / 3 for a, _ in sch)
        joined = " ".join(t for _, t in sch)
        if r["src"]:
            ratios.append(len(joined) / len(r["src"]))
        if known_fn is not None:
            if content_fn is None:
                from build_prefix import content as content_fn
            known = known_fn(r["sid"])
            for a, t in sch:
                c = content_fn(t, r["tgt"])
                j = next((k for k, s in enumerate(known, 1) if c & s), None)
                if j is not None:
                    defs.append(a - j)
    return {
        "n": len(rows), "empty_share": _mean(empties), "dump_share": dump / big if big else None,
        "first_after_median": _pct(firsts, 0.5), "first_after_p95": _pct(firsts, 0.95),
        "deferral_median": _pct(defs, 0.5), "deferral_max": max(defs) if defs else None,
        "len_ratio_median": _pct(ratios, 0.5), "len_ratio_p5": _pct(ratios, 0.05),
        "len_ratio_p95": _pct(ratios, 0.95), "gated_out": gated_out,
        "piece_len_median": _pct(lens, 0.5), "short_share": short / short_tot if short_tot else None,
        "pieces_per_word": _pct(per_word, 0.5), "single_share": single / len(rows) if rows else None,
        "mid_closed_share": mid_closed / mid_tot if mid_tot else None,
        "first_bare_subj": bare / len(rows) if rows else None,
        "first_after_frac": _pct(first_frac, 0.5),
        "point_empty_share": pt_empty / pt_tot if pt_tot else None,
        "regret_share": regret / pt_tot if pt_tot else None,
        "est_lag_s": _mean(lags),
    }


def stats_old(rows):
    """Same metrics for old phrase-level rows (see module docstring for the approximations)."""
    empt = tot = dump = multi = 0
    firsts, ratios = [], []
    for r in rows:
        adds = r["adds"]
        P = len(adds)
        N = len(r["src"].split())
        if not P:
            continue
        tot += P
        empt += sum(1 for a in adds if not a.strip())
        if P >= 2:
            multi += 1
            dump += all(not a.strip() for a in adds[:-1])
        i0 = next((i for i, a in enumerate(adds) if a.strip()), None)
        if i0 is not None:
            firsts.append(round(i0 * N / P))
        if r["src"]:
            ratios.append(len(" ".join(a for a in adds if a.strip())) / len(r["src"]))
    return {
        "n": len(rows), "empty_share": empt / tot if tot else None, "dump_share": dump / multi if multi else None,
        "first_after_median": _pct(firsts, 0.5), "first_after_p95": _pct(firsts, 0.95),
        "deferral_median": None, "deferral_max": None,
        "len_ratio_median": _pct(ratios, 0.5), "len_ratio_p5": _pct(ratios, 0.05),
        "len_ratio_p95": _pct(ratios, 0.95), "gated_out": 0,
        "piece_len_median": None, "short_share": None, "pieces_per_word": None, "single_share": None,
    }


def make_wait_labels(sids, sents, hy, hy_kiwi, kiwi_floor, why_of=None):
    """Wait labels for rescued sids: hear the whole source, then emit the whole translation (the reference, or the
    Hy translation when there is none). A source with 2-3 clauses whose translation has as many sentences is split
    clause by clause. -> (labels, dropped) where dropped lists no-reference sids whose Hy kiwi is under the floor."""
    labels, dropped = [], []
    for sid in sids:
        s = sents[sid]
        why = (why_of or {}).get(sid, "comet")
        full = (s.get("ref") or "").strip()
        if not full:
            full = (hy.get(sid) or [""])[-1].strip()
            k = (hy_kiwi or {}).get(sid)
            if kiwi_floor is not None and k is not None and k < kiwi_floor:
                dropped.append({"sid": sid, "why": "hy_kiwi_floor", "kiwi_hy": k})
                continue
        if not full:
            dropped.append({"sid": sid, "why": "no_translation"})
            continue
        toks = words(s["src"], s["lang"])
        sched = [[len(toks), full]]
        ends = [e for e in clause_ends(toks, s["lang"]) if _PUNCT_TOKEN.search(toks[e - 1])]  # sentence-final only
        parts = [x.strip() for x in _SENT_SPLIT.split(full) if x.strip()]
        if (2 <= len(ends) <= 3 and len(parts) == len(ends) and ends[-1] == len(toks)
                and all(len(x.split()) >= 2 and not _ABBREV_END.search(x) for x in parts)):
            sched = [[e, t] for e, t in zip(ends, parts)]
        labels.append({"sid": sid, "lang": s["lang"], "tgt": s["tgt"], "schedule": sched, "rescue": "wait", "why": why})
    return labels, dropped


def gate(rows, sents, hy, comet_min, kiwi_margin, extra_hy_sids=()):
    """Score rows with COMET models -> (surviving rows with scores attached, dropped rows, Hy kiwi by sid).
    `extra_hy_sids`: no-reference sids without a label (rescue candidates) whose Hy translation is scored too."""
    from comet import download_model, load_from_checkpoint

    with_ref = [r for r in rows if sents[r["sid"]].get("ref")]
    no_ref = [r for r in rows if not sents[r["sid"]].get("ref")]
    joined = lambda r: join_texts([t for _, t in r["schedule"]])
    keep, hy_kiwi = [], {}
    if with_ref:
        m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": sents[r["sid"]]["src"], "mt": joined(r), "ref": sents[r["sid"]]["ref"]} for r in with_ref]
        for r, s in zip(with_ref, m.predict(data, batch_size=64, gpus=1, progress_bar=False).scores):
            r["comet"] = round(float(s), 4)
            if s >= comet_min:
                keep.append(r)
    extra = [x for x in extra_hy_sids if not sents[x].get("ref") and hy.get(x)]
    if no_ref or extra:
        m = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        data = [{"src": sents[r["sid"]]["src"], "mt": joined(r)} for r in no_ref] + [
            {"src": sents[r["sid"]]["src"], "mt": hy[r["sid"]][-1]} for r in no_ref] + [
            {"src": sents[x]["src"], "mt": hy[x][-1]} for x in extra]
        sc = m.predict(data, batch_size=64, gpus=1, progress_bar=False).scores
        n = len(no_ref)
        for r, k, kh in zip(no_ref, sc[:n], sc[n:2 * n]):
            r["kiwi"], r["kiwi_hy"] = round(float(k), 4), round(float(kh), 4)
            hy_kiwi[r["sid"]] = r["kiwi_hy"]
            if k >= kh - kiwi_margin:
                keep.append(r)
        for x, kh in zip(extra, sc[2 * n:]):
            hy_kiwi[x] = round(float(kh), 4)
    kept_ids = {r["sid"] for r in keep}
    return keep, [r for r in rows if r["sid"] not in kept_ids], hy_kiwi


_COLS = ["n", "empty_share", "dump_share", "first_after_median", "first_after_p95", "deferral_median",
         "deferral_max", "len_ratio_median", "len_ratio_p5", "len_ratio_p95", "gated_out",
         "piece_len_median", "short_share", "pieces_per_word", "single_share",
         "mid_closed_share", "first_bare_subj", "first_after_frac", "point_empty_share", "regret_share", "est_lag_s"]


def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def rescue_report(kept, waits, sents, judge_rows):
    """Per direction: wait-label count/share by reason, judge_fail share (first judge), retried share (gate retry,
    attempts >= 2) and relabeled share (judge relabel, judge_attempts >= 2)."""
    out = {}
    key = lambda sid: f"{sents[sid]['lang']}->{sents[sid]['tgt']}"
    for r in kept:
        d = out.setdefault(key(r["sid"]), {"kept": 0, "wait": 0, "by_why": {}, "retried": 0, "judge_fail": 0, "judged": 0})
        d["kept"] += 1
        d["retried"] += r.get("attempts", 1) >= 2
        d["relabeled"] = d.get("relabeled", 0) + (r.get("judge_attempts", 1) >= 2)
    for r in waits:
        d = out.setdefault(key(r["sid"]), {"kept": 0, "wait": 0, "by_why": {}, "retried": 0, "judge_fail": 0, "judged": 0})
        d["wait"] += 1
        d["by_why"][r["why"]] = d["by_why"].get(r["why"], 0) + 1
    for j in judge_rows:
        if j["sid"] in sents:
            d = out.setdefault(key(j["sid"]), {"kept": 0, "wait": 0, "by_why": {}, "retried": 0, "judge_fail": 0, "judged": 0})
            d["judged"] += 1
            d["judge_fail"] += not j.get("pass") and not j.get("unjudged")
    for d in out.values():
        tot = d["kept"] + d["wait"]
        d["rescued_wait_share"] = d["wait"] / tot if tot else None
        d["retried_share"] = d["retried"] / d["kept"] if d["kept"] else None
        d["relabeled_share"] = d.get("relabeled", 0) / d["kept"] if d["kept"] else None
        d["judge_fail_share"] = d["judge_fail"] / d["judged"] if d["judged"] else None
    return out


def print_rescue(kept, waits, sents, judge_rows):
    print("\n== rescue / judge")
    for d, m in sorted(rescue_report(kept, waits, sents, judge_rows).items()):
        print(d.ljust(8), f"kept {m['kept']}", f"wait {m['wait']} ({_fmt(m['rescued_wait_share'])}) {m['by_why']}",
              f"retried {_fmt(m['retried_share'])}", f"relabeled {_fmt(m['relabeled_share'])}", f"judge_fail {_fmt(m['judge_fail_share'])}")


def print_table(title, by_dir):
    print(f"\n== {title}")
    print("direction".ljust(8), *[c.rjust(18) for c in _COLS])
    for d in sorted(by_dir):
        print(d.ljust(8), *[_fmt(by_dir[d].get(c)).rjust(18) for c in _COLS])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--hy", required=True)
    ap.add_argument("--old")
    ap.add_argument("--out-gated", required=True)
    ap.add_argument("--skip-completeness", action="store_true")
    ap.add_argument("--comet_min", type=float, default=0.70)
    ap.add_argument("--kiwi_margin", type=float, default=0.08)
    ap.add_argument("--rescue", help="compose_judged.py rescue list: these sids get wait labels too")
    ap.add_argument("--judge", help="first judge file: judge_fail share per direction is reported from it")
    ap.add_argument("--kiwi-floor-pct", type=float, default=10.0,
                    help="no-reference wait labels whose Hy kiwi is under this percentile of all Hy kiwi scores are dropped")
    ap.add_argument("--out-dropped", help="jsonl of sids that got no label at all (under the kiwi floor / no translation)")
    ap.add_argument("--word-times", help="word_times.jsonl: with it the table reports est_lag_s (first recording per sentence)")
    a = ap.parse_args(argv)

    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    sents = {r["sid"]: r for r in load(a.sentences)}
    hy = {r["sid"]: r["prefixes"] for r in load(a.hy)}
    rows = [r for r in load(a.schedule) if r["sid"] in sents]
    rescue = {r["sid"]: r.get("why", "fail") for r in load(a.rescue)} if a.rescue else {}
    rescue = {sid: why for sid, why in rescue.items() if sid in sents}
    if a.skip_completeness:
        kept, dropped_rows, hy_kiwi = rows, [], {}
    else:
        kept, dropped_rows, hy_kiwi = gate(rows, sents, hy, a.comet_min, a.kiwi_margin, extra_hy_sids=list(rescue))
    kept_ids = {r["sid"] for r in kept}

    dropped = defaultdict(int)
    for r in rows:
        if r["sid"] not in kept_ids:
            dropped[f"{r['lang']}->{r['tgt']}"] += 1
    print("gate dropped:", dict(dropped) or "none", f"({len(kept)}/{len(rows)} kept)")

    why_of = {r["sid"]: "comet" for r in dropped_rows}
    why_of.update(rescue)
    wait_sids = [sid for sid in why_of if sid not in kept_ids]
    floor = _pct(sorted(hy_kiwi.values()), a.kiwi_floor_pct / 100) if hy_kiwi else None
    waits, no_label = make_wait_labels(wait_sids, sents, hy, hy_kiwi, floor, why_of)
    print(f"wait labels: {len(waits)} (kiwi floor {floor if floor is None else round(floor, 4)}), no label at all: {len(no_label)}")
    with open(a.out_gated, "w", encoding="utf-8") as f:
        for r in kept + waits:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.out_dropped:
        with open(a.out_dropped, "w", encoding="utf-8") as f:
            for r in no_label:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from hy_word_prefixes import known_sets

    groups = defaultdict(list)
    for r in kept:
        s = sents[r["sid"]]
        groups[f"{s['lang']}->{s['tgt']}"].append({**r, "src": s["src"], "lang": s["lang"], "tgt": s["tgt"], "ref": s.get("ref")})
    known_fn = lambda sid: known_sets(hy[sid], sents[sid]["tgt"])
    wt = None
    if a.word_times:
        by_audio = {r["audio"]: r["words"] for r in load(a.word_times)}
        wt = {sid: by_audio[s["recs"][0]["audio"]] for sid, s in sents.items() if s.get("recs") and s["recs"][0]["audio"] in by_audio}
    new = {d: stats_for(g, words, known_fn, gated_out=dropped[d], word_times=wt) for d, g in groups.items()}
    for d in dropped:  # a direction the gate removed entirely still gets a (zero) row
        if d not in new:
            new[d] = stats_for([], words, gated_out=dropped[d])
    print_table("new schedule corpus", new)
    print_rescue(kept, waits, sents, load(a.judge) if a.judge else [])
    if a.old:
        og = defaultdict(list)
        for r in load(a.old):
            og[f"{r['lang']}->{r['tgt']}"].append(r)
        print_table("old corpus (phrase-level approximation)", {d: stats_old(g) for d, g in og.items()})
    print("SCHEDULE_STATS_DONE")


if __name__ == "__main__":
    main()
