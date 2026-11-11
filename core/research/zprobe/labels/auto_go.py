"""Unattended GO decision for the prefix14 chain: the hygiene checks the controller would do by hand on the
stats table, applied to the gated schedule file. Exit 0 = GO (the chain touches $OUT/GO), 1 = stop with reasons.

Checks (per direction = lang-tgt):
  kept        gated rows >= --min-kept and >= --min-kept-share of the labeled rows
  dump_share  share of sentences (>= 4 words) whose every item sits after 2/3 of the source: en-ko <= --dump-enko,
              other directions <= --dump-other (en->ko is where the old labels dumped at the end)
  empty       empty_share of the two directions within --empty-ratio of each other (one direction must not be
              taught to wait far more than the other)
  first_after first_after_median <= --first-after (the first piece must not routinely wait past this many words)

usage: PYTHONPATH=. python core/research/zprobe/labels/auto_go.py --gated <gated.jsonl> --labeled <schedule.jsonl>"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import words  # noqa: E402
from schedule_stats import stats_for  # noqa: E402


def check(rows, n_labeled, a):
    by = {}
    for r in rows:
        by.setdefault(f"{r['lang']}-{r['tgt']}", []).append(r)
    reasons = []
    if len(rows) < a.min_kept:
        reasons.append(f"kept {len(rows)} < {a.min_kept}")
    if n_labeled and len(rows) < a.min_kept_share * n_labeled:
        reasons.append(f"kept {len(rows)} < {a.min_kept_share:.0%} of labeled {n_labeled}")
    table = {d: stats_for(v, words) for d, v in by.items()}
    for d, m in table.items():
        lim = a.dump_enko if d == "en-ko" else a.dump_other
        if m["dump_share"] is not None and m["dump_share"] > lim:
            reasons.append(f"{d} dump_share {m['dump_share']:.3f} > {lim}")
        if m["first_after_median"] is not None and m["first_after_median"] > a.first_after:
            reasons.append(f"{d} first_after_median {m['first_after_median']} > {a.first_after}")
    es = {d: m["empty_share"] for d, m in table.items() if m["empty_share"] is not None}
    if len(es) >= 2 and max(es.values()) > a.empty_ratio * max(min(es.values()), 0.05):  # floor: 0 vs 0.08 is fine
        reasons.append(f"empty_share ratio {max(es.values()) / max(min(es.values()), 0.05):.2f} > {a.empty_ratio}: {es}")
    return table, reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--min-kept", type=int, default=4000)
    ap.add_argument("--min-kept-share", type=float, default=0.5)
    ap.add_argument("--dump-enko", type=float, default=0.05)
    ap.add_argument("--dump-other", type=float, default=0.10)
    ap.add_argument("--empty-ratio", type=float, default=2.0)
    ap.add_argument("--first-after", type=float, default=6)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(a.gated, encoding="utf-8") if l.strip()]
    n_labeled = sum(1 for l in open(a.labeled, encoding="utf-8") if l.strip()) if os.path.exists(a.labeled) else 0
    table, reasons = check(rows, n_labeled, a)
    for d, m in sorted(table.items()):
        print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    if reasons:
        print("AUTO_GO_NO:", "; ".join(reasons))
        return 1
    print(f"AUTO_GO_YES kept={len(rows)} labeled={n_labeled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
