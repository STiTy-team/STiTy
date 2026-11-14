"""Unattended GO decision for the prefix14 chain: the hygiene checks the controller would do by hand on the
stats table, applied to the gated schedule file. Exit 0 = GO (the chain touches $OUT/GO), 1 = stop with reasons.

Checks (per direction = lang-tgt):
  kept        gated rows >= --min-kept and >= --min-kept-share of the labeled rows
  dump_share  share of sentences (>= 4 words) whose every item sits after 2/3 of the source: en-ko <= --dump-enko,
              other directions <= --dump-other (en->ko is where the old labels dumped at the end)
  empty       empty_share of the two directions within --empty-ratio of each other (one direction must not be
              taught to wait far more than the other)
  first_after first_after_median <= --first-after (the first piece must not routinely wait past this many words; 8 = a stop, not a target)
  rescued     wait labels (rescue == "wait", counted apart from the table) <= --rescued-max of the corpus

usage: PYTHONPATH=. python core/research/zprobe/labels/auto_go.py --gated <gated.jsonl> --labeled <schedule.jsonl> \
         --sentences <sentences.jsonl>     (the labeler's rows carry no src; it is merged in by sid)"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import words  # noqa: E402
from schedule_stats import stats_for  # noqa: E402


def check(rows, n_labeled, a, word_times=None):
    """rows may include wait labels (rescue == "wait"): they are counted as rescued, not in the corpus table."""
    waits = [r for r in rows if r.get("rescue") == "wait"]
    rows = [r for r in rows if r.get("rescue") != "wait"]
    by = {}
    for r in rows:
        by.setdefault(f"{r['lang']}-{r['tgt']}", []).append(r)
    reasons = []
    rescued_max = getattr(a, "rescued_max", 0.25)
    if waits and len(waits) > rescued_max * (len(rows) + len(waits)):
        reasons.append(f"rescued wait labels {len(waits)}/{len(rows) + len(waits)} = {len(waits) / (len(rows) + len(waits)):.3f} > {rescued_max}")
    if len(rows) < a.min_kept:
        reasons.append(f"kept {len(rows)} < {a.min_kept}")
    if n_labeled and len(rows) < a.min_kept_share * n_labeled:
        reasons.append(f"kept {len(rows)} < {a.min_kept_share:.0%} of labeled {n_labeled}")
    table = {d: stats_for(v, words, word_times=word_times) for d, v in by.items()}
    for d, m in table.items():
        lim = a.dump_enko if d == "en-ko" else a.dump_other
        if m["dump_share"] is not None and m["dump_share"] > lim:
            reasons.append(f"{d} dump_share {m['dump_share']:.3f} > {lim}")
        if m["first_after_median"] is not None and m["first_after_median"] > a.first_after:
            reasons.append(f"{d} first_after_median {m['first_after_median']} > {a.first_after}")
        # prefix16 (open pieces): closing mid-source, bare-subject openers, too few waits, too many regrets, lag
        for key, attr, above, default in (("mid_closed_share", "mid_closed", True, 0.03), ("first_bare_subj", "first_bare", True, 0.05),
                                          ("point_empty_share", "empty_min", False, 0.0), ("regret_share", "regret_max", True, 0.30),
                                          ("est_lag_s", "lag_max", True, 0.0)):
            lim = getattr(a, attr, default)
            v = m.get(key)
            if v is None:
                if key == "est_lag_s" and lim:
                    reasons.append(f"{d} est_lag_s unavailable (pass --word-times) while --lag-max {lim} is set")
                continue
            if above and v > lim:
                reasons.append(f"{d} {key} {v:.3f} > {lim}")
            if not above and v < lim:
                reasons.append(f"{d} {key} {v:.3f} < {lim}")
    es = {d: m["empty_share"] for d, m in table.items() if m["empty_share"] is not None}
    if len(es) >= 2 and max(es.values()) > a.empty_ratio * max(min(es.values()), 0.05):  # floor: 0 vs 0.08 is fine
        reasons.append(f"empty_share ratio {max(es.values()) / max(min(es.values()), 0.05):.2f} > {a.empty_ratio}: {es}")
    return table, reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--min-kept", type=int, default=4000)
    ap.add_argument("--min-kept-share", type=float, default=0.5)
    ap.add_argument("--dump-enko", type=float, default=0.15)
    ap.add_argument("--rescued-max", type=float, default=0.25, help="wait labels (rescued) may be at most this share of the corpus")
    ap.add_argument("--dump-other", type=float, default=0.10)
    ap.add_argument("--empty-ratio", type=float, default=2.0)
    ap.add_argument("--first-after", type=float, default=8)
    ap.add_argument("--mid-closed", type=float, default=0.03, help="non-final pieces that close a sentence, at most")
    ap.add_argument("--first-bare", type=float, default=0.05, help="sentences whose first piece is a bare subject, at most")
    ap.add_argument("--empty-min", type=float, default=0.0, help="empty (wait) items among all items, at least (0 = off; prefix16 chain passes 0.05)")
    ap.add_argument("--regret-max", type=float, default=0.30, help="sentences that went through a regret round, at most")
    ap.add_argument("--lag-max", type=float, default=0.0, help="est_lag_s at most; needs --word-times (0 = off; the prefix16 chain passes 1.6)")
    ap.add_argument("--word-times", help="word_times.jsonl; est_lag_s is measured on each sentence's first recording")
    a = ap.parse_args(argv)
    src = {}
    for l in open(a.sentences, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            src[r["sid"]] = r["src"]
    rows = [json.loads(l) for l in open(a.gated, encoding="utf-8") if l.strip()]
    missing = [r["sid"] for r in rows if r["sid"] not in src]
    if missing:
        print(f"AUTO_GO_NO: {len(missing)} gated sids absent from sentences (e.g. {missing[:3]})")
        return 1
    for r in rows:
        r["src"] = src[r["sid"]]
    wt = None
    if a.word_times:
        first_audio = {}
        for l in open(a.sentences, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("recs"):
                    first_audio[r["sid"]] = r["recs"][0]["audio"]
        by_audio = {}
        for l in open(a.word_times, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                by_audio[r["audio"]] = r["words"]
        wt = {sid: by_audio[au] for sid, au in first_audio.items() if au in by_audio}
    n_labeled = sum(1 for l in open(a.labeled, encoding="utf-8") if l.strip()) if os.path.exists(a.labeled) else 0
    table, reasons = check(rows, n_labeled, a, word_times=wt)
    for d, m in sorted(table.items()):
        print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    if reasons:
        print("AUTO_GO_NO:", "; ".join(reasons))
        return 1
    n_wait = sum(1 for r in rows if r.get("rescue") == "wait")
    print(f"AUTO_GO_YES kept={len(rows) - n_wait} wait={n_wait} labeled={n_labeled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
