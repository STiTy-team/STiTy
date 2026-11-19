"""Sweep decision settings over replay.py outputs and print one table.

For segment rows the stored raw span scores are re-decided with each (temperature, stay_prior,
conf); for timeline rows the stored steps (known at the decision) are re-smoothed with each
(temperature, p_stay) and re-decided with each conf. `alphabet` on the commit text is the baseline.

usage: python core/research/lid_module/sweep.py EVAL_DIR > grid.md
"""

import glob
import itertools
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.getcwd())

from core.components.langid.audio.decide import decide, normalise, smooth, with_prior  # noqa: E402
from core.utils import langs as langs_mod  # noqa: E402

TEMPS = (1.0, 3.0, 10.0)
STAYS = (0.5, 0.7, 0.9)
PSTAYS = (0.8, 0.9, 0.95, 0.99)
CONFS = (0.5, 0.6, 0.7, 0.8, 0.9)
HALF = 0.5  # half of the shortest timeline window (1 s)


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def scaled(scores, temp):
    return None if scores is None else {k: v * temp for k, v in scores.items()}


def segment_acc(rows, cands, temp, stay, conf):
    hit = n = 0
    prev_conv, previous = None, None
    for r in rows:
        if r["conv"] != prev_conv:
            prev_conv, previous = r["conv"], None
        raw = scaled(r["raw"], temp)
        probs = None if raw is None else normalise(with_prior(raw, previous, stay), cands)
        got = decide(probs, previous, conf)
        previous = got or previous
        if r["truth"]:
            n += 1
            hit += got == r["truth"]
    return hit / n if n else None


def timeline_acc(rows, cands, temp, p_stay, conf):
    hit = n = 0
    prev_conv, previous = None, None
    for r in rows:
        if r["conv"] != prev_conv:
            prev_conv, previous = r["conv"], None
        steps = r["steps"]
        probs = None
        if steps:
            posts = smooth([scaled(s, temp) for _, s in steps], cands, p_stay)
            lo, hi = r["span"]
            inside = [i for i, (t, s) in enumerate(steps) if lo <= t - HALF <= hi and s]
            if not inside:
                inside = [min(range(len(steps)), key=lambda i: abs(steps[i][0] - HALF - hi))]
            mass = {c: sum(posts[i][c] for i in inside) for c in cands}
            total = sum(mass.values()) or 1.0
            probs = {c: v / total for c, v in mass.items()}
        got = decide(probs, previous, conf)
        previous = got or previous
        if r["truth"]:
            n += 1
            hit += got == r["truth"]
    return hit / n if n else None


def alphabet_acc(rows, cands):
    hits = [
        langs_mod.detect_by_alphabet(r["text"], langs=cands, threshold=langs_mod.ALPHABET_THRESHOLD)
        == r["truth"]
        for r in rows
        if r["truth"]
    ]
    return sum(hits) / len(hits)


def main(eval_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(eval_dir, "grid", "*.jsonl"))):
        dataset, mode, scorer = os.path.basename(path)[:-6].split("__")
        rows = load(path)
        if not rows:
            continue
        cands = tuple(json.load(open(os.path.join(eval_dir, "langs.json")))[dataset])
        if mode == "segment":
            grid = {
                (t, s, c): segment_acc(rows, cands, t, s, c)
                for t, s, c in itertools.product(TEMPS, STAYS, CONFS)
            }
        else:
            grid = {
                (t, p, c): timeline_acc(rows, cands, t, p, c)
                for t, p, c in itertools.product(TEMPS, PSTAYS, CONFS)
            }
        src = sum(r["src"] == "aligner" for r in rows) / len(rows)
        cells[(dataset, mode, scorer)] = (grid, alphabet_acc(rows, cands), src)

    datasets = sorted({d for d, _, _ in cells})
    print(
        "## Per dataset — default settings (T=1, stay/p_stay 0.7/0.9, conf 0.7) and best setting\n"
    )
    print("| dataset | mode | scorer | default | best | best setting | alphabet | aligner spans |")
    print("|---|---|---|---|---|---|---|---|")
    for (d, m, s), (grid, alpha, src) in sorted(cells.items()):
        default = grid[(1.0, 0.7 if m == "segment" else 0.9, 0.7)]
        best_key = max(grid, key=lambda k: grid[k])
        print(
            f"| {d} | {m} | {s} | {default:.3f} | {grid[best_key]:.3f} | {best_key} | {alpha:.3f} | {src:.2f} |"
        )

    print("\n## One setting per (mode, scorer), chosen by mean accuracy over datasets\n")
    print("| mode | scorer | setting | mean | " + " | ".join(datasets) + " |")
    print("|---|---|---|---|" + "---|" * len(datasets))
    ranking = []
    for m, s in sorted({(m, s) for _, m, s in cells}):
        have = [d for d in datasets if (d, m, s) in cells]
        keys = set.intersection(*(set(cells[(d, m, s)][0]) for d in have))
        best = max(keys, key=lambda k: mean(cells[(d, m, s)][0][k] for d in have))
        accs = {d: cells[(d, m, s)][0][best] for d in have}
        ranking.append((mean(accs.values()), m, s, best, len(have)))
        cols = " | ".join(f"{accs[d]:.3f}" if d in accs else "-" for d in datasets)
        print(f"| {m} | {s} | {best} | {mean(accs.values()):.3f} | {cols} |")
    alpha = {d: next(v[1] for (dd, _, _), v in cells.items() if dd == d) for d in datasets}
    print(
        f"| baseline | alphabet | - | {mean(alpha.values()):.3f} | "
        + " | ".join(f"{alpha[d]:.3f}" for d in datasets)
        + " |"
    )
    print("\n## Ranking (complete cells only)\n")
    for acc, m, s, best, n in sorted(ranking, reverse=True):
        print(f"- {m} {s} {best}: {acc:.3f} over {n} datasets")
    json.dump(
        [
            dict(mean=a, mode=m, scorer=s, setting=list(b), datasets=n)
            for a, m, s, b, n in sorted(ranking, reverse=True)
        ],
        open(os.path.join(eval_dir, "ranking.json"), "w"),
    )


if __name__ == "__main__":
    main(sys.argv[1])
