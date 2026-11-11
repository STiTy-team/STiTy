"""Completeness gate + label hygiene table for emission-schedule labels.

1. Gate: keep a schedule only if its joined text is a complete translation.
   - ref present: COMET-da(src, joined, ref) >= --comet_min (0.70)
   - ref null:    cometkiwi(joined) >= cometkiwi(Hy full translation) - --kiwi_margin (0.08)
   Survivors (with "comet" or "kiwi"/"kiwi_hy" added) go to --out-gated. --skip-completeness copies every row
   (CPU tests / stats-only check; comet is imported lazily so it is not needed then).
2. Hygiene table per direction lang->tgt: n, empty_share, dump_share, first_after_median/p95, deferral_median/max,
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
from schedule_gate import words  # noqa: E402


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


def stats_for(rows, tokens_fn, known_fn=None, content_fn=None, gated_out=0):
    """rows: dicts with sid, lang, tgt, src, schedule [[after, text], ...]. Returns the metric dict."""
    empties, firsts, defs, ratios = [], [], [], []
    dump = big = 0
    for r in rows:
        K = len(tokens_fn(r["src"], r["lang"]))
        sch = [(int(a), t) for a, t in r["schedule"]]
        if K and sch:
            empties.append(1 - len({a for a, _ in sch}) / K)
            firsts.append(min(a for a, _ in sch))
        if K >= 4:
            big += 1
            dump += bool(sch) and all(a > 2 * K / 3 for a, _ in sch)
        joined = " ".join(t for _, t in sch if t.strip())
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
    }


def gate(rows, sents, hy, comet_min, kiwi_margin):
    """Score rows with COMET models; returns the surviving rows with scores attached."""
    from comet import download_model, load_from_checkpoint

    with_ref = [r for r in rows if sents[r["sid"]].get("ref")]
    no_ref = [r for r in rows if not sents[r["sid"]].get("ref")]
    joined = lambda r: " ".join(t.strip() for _, t in r["schedule"] if t.strip())
    keep = []
    if with_ref:
        m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": sents[r["sid"]]["src"], "mt": joined(r), "ref": sents[r["sid"]]["ref"]} for r in with_ref]
        for r, s in zip(with_ref, m.predict(data, batch_size=64, gpus=1, progress_bar=False).scores):
            r["comet"] = round(float(s), 4)
            if s >= comet_min:
                keep.append(r)
    if no_ref:
        m = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))
        data = [{"src": sents[r["sid"]]["src"], "mt": joined(r)} for r in no_ref] + [
            {"src": sents[r["sid"]]["src"], "mt": hy[r["sid"]][-1]} for r in no_ref]
        sc = m.predict(data, batch_size=64, gpus=1, progress_bar=False).scores
        n = len(no_ref)
        for r, k, kh in zip(no_ref, sc[:n], sc[n:]):
            r["kiwi"], r["kiwi_hy"] = round(float(k), 4), round(float(kh), 4)
            if k >= kh - kiwi_margin:
                keep.append(r)
    return keep


_COLS = ["n", "empty_share", "dump_share", "first_after_median", "first_after_p95", "deferral_median",
         "deferral_max", "len_ratio_median", "len_ratio_p5", "len_ratio_p95", "gated_out"]


def _fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_table(title, by_dir):
    print(f"\n== {title}")
    print("direction".ljust(8), *[c.rjust(18) for c in _COLS])
    for d in sorted(by_dir):
        print(d.ljust(8), *[_fmt(by_dir[d][c]).rjust(18) for c in _COLS])


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
    a = ap.parse_args(argv)

    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    sents = {r["sid"]: r for r in load(a.sentences)}
    hy = {r["sid"]: r["prefixes"] for r in load(a.hy)}
    rows = [r for r in load(a.schedule) if r["sid"] in sents]
    kept = rows if a.skip_completeness else gate(rows, sents, hy, a.comet_min, a.kiwi_margin)
    kept_ids = {r["sid"] for r in kept}
    with open(a.out_gated, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dropped = defaultdict(int)
    for r in rows:
        if r["sid"] not in kept_ids:
            dropped[f"{r['lang']}->{r['tgt']}"] += 1
    print("gate dropped:", dict(dropped) or "none", f"({len(kept)}/{len(rows)} kept)")

    from hy_word_prefixes import known_sets

    groups = defaultdict(list)
    for r in kept:
        s = sents[r["sid"]]
        groups[f"{s['lang']}->{s['tgt']}"].append({**r, "src": s["src"], "lang": s["lang"], "tgt": s["tgt"], "ref": s.get("ref")})
    known_fn = lambda sid: known_sets(hy[sid], sents[sid]["tgt"])
    new = {d: stats_for(g, words, known_fn, gated_out=dropped[d]) for d, g in groups.items()}
    print_table("new schedule corpus", new)
    if a.old:
        og = defaultdict(list)
        for r in load(a.old):
            og[f"{r['lang']}->{r['tgt']}"].append(r)
        print_table("old corpus (phrase-level approximation)", {d: stats_old(g) for d, g in og.items()})
    print("SCHEDULE_STATS_DONE")


if __name__ == "__main__":
    main()
