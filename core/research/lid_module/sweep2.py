"""Tune the audio langid offline on replay2.py outputs: settings are picked on the FLEURS mixes
(coordinate search, two passes) and northstar is only evaluated. Then each improvement is turned
off one at a time from the picked setting (ablation).

usage: python core/research/lid_module/sweep2.py EVAL_DIR > grid2.md
"""

import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.getcwd())

from core.components.langid.audio.decide import (  # noqa: E402
    combine,
    decide,
    normalise,
    smooth,
    span_mass,
    with_prior,
)

TRAIN = ("ko+en", "ja+en", "es+en", "ko+ja+en")
TEST = ("northstar",)
SCORERS = ("whisper-token", "ecapa", "mms")
SUBSETS = [(s,) for s in SCORERS] + [
    ("whisper-token", "ecapa"),
    ("whisper-token", "mms"),
    ("ecapa", "mms"),
    SCORERS,
]
TIMELINE_SPACE = {
    "scorers": SUBSETS,
    "combine": ["mean", "max"],
    "windows": [("1.0", "2.0", "4.0"), ("1.0", "2.0"), ("2.0", "4.0"), ("1.0",), ("2.0",)],
    "temp": [1.0, 3.0, 10.0],
    "p_stay": [0.8, 0.9, 0.95, 0.99],
    "p_gap": [None, 0.5, 0.7],
    "read": ["mean", "tail", "end"],
    "conf": [0.5, 0.7],
}
TIMELINE_START = {
    "scorers": ("whisper-token",),
    "combine": "mean",
    "windows": ("1.0", "2.0", "4.0"),
    "temp": 10.0,
    "p_stay": 0.8,
    "p_gap": None,
    "read": "mean",
    "conf": 0.5,
}
SEGMENT_SPACE = {
    "scorers": SUBSETS,
    "combine": ["mean", "max"],
    "temp": [1.0, 3.0, 10.0],
    "stay": [0.5, 0.7, 0.9],
    "conf": [0.5, 0.6, 0.7, 0.8],
}
SEGMENT_START = {
    "scorers": ("whisper-token",),
    "combine": "mean",
    "temp": 1.0,
    "stay": 0.7,
    "conf": 0.5,
}


def load(path):
    return (
        [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []
    )


def step_scores(raw, cands, s):
    """One step's log-scores from its raw per-scorer, per-window scores (component logic)."""
    merged = []
    for w in s["windows"]:
        per = {n: raw[n][w] for n in s["scorers"] if n in raw and w in raw[n]}
        c = combine(per, cands, s["combine"]) if per else None
        if c:
            merged.append(c)
    if not merged:
        return None
    return {c: s["temp"] * sum(m[c] for m in merged) / len(merged) for c in cands}


def timeline_decisions(rows, cands, s, centered):
    out, prev_conv, previous = [], None, None
    half = 0.0 if centered else min(float(w) for w in s["windows"]) / 2
    cache = {}
    for r in rows:
        if r["conv"] != prev_conv:
            prev_conv, previous = r["conv"], None
        steps = []
        for t, silent, raw in r["steps"]:
            key = (r["conv"], t, bool(raw))
            if key not in cache or t > r["decision"] - 2.5:  # late centered steps are provisional
                cache[key] = step_scores(raw, cands, s) if raw else None
            steps.append((t, silent, cache[key]))
        probs = None
        if steps:
            gaps = [False] + [q for _, q, _ in steps[:-1]]
            posts = smooth([x for _, _, x in steps], cands, s["p_stay"], s["p_gap"], gaps)
            probs = span_mass(
                [(t, x) for t, _, x in steps], posts, tuple(r["span"]), half, cands, s["read"], 1.0
            )
        got = decide(probs, previous, s["conf"])
        previous = got or previous
        out.append((r, got))
    return out


def segment_decisions(rows, cands, s):
    out, prev_conv, previous = [], None, None
    for r in rows:
        if r["conv"] != prev_conv:
            prev_conv, previous = r["conv"], None
        probs = None
        if r["raw"]:
            per = {n: r["raw"][n] for n in s["scorers"] if n in r["raw"]}
            merged = combine(per, cands, s["combine"]) if per else None
            if merged:
                scaled = {c: v * s["temp"] for c, v in merged.items()}
                probs = normalise(with_prior(scaled, previous, s["stay"]), cands)
        got = decide(probs, previous, s["conf"])
        previous = got or previous
        out.append((r, got))
    return out


def accuracy(decisions):
    m = [(r, g) for r, g in decisions if r["truth"]]
    return sum(g == r["truth"] for r, g in m) / len(m) if m else None


def switch_error(decisions):
    """Error rate on the first commit after a true language change."""
    wrong = n = 0
    prev_conv, prev_truth = None, None
    for r, g in decisions:
        if r["conv"] != prev_conv:
            prev_conv, prev_truth = r["conv"], None
        if not r["truth"]:
            continue
        if prev_truth is not None and r["truth"] != prev_truth:
            n += 1
            wrong += g != r["truth"]
        prev_truth = r["truth"]
    return wrong / n if n else None


def tune(evaluate, space, start):
    best = dict(start)
    score = evaluate(best)
    for _ in range(2):
        for key, options in space.items():
            for value in options:
                trial = {**best, key: value}
                acc = evaluate(trial)
                if acc > score + 1e-9:
                    best, score = trial, acc
    return best, score


def main(eval_dir):
    langs = json.load(open(os.path.join(eval_dir, "langs.json")))
    data = {
        (kind, d): load(os.path.join(eval_dir, "grid2", f"{d}__{kind}.jsonl"))
        for kind in ("segment", "centered", "trailing")
        for d in TRAIN + TEST
    }

    def run(kind, d, s):
        rows, cands = data[(kind, d)], tuple(langs[d])
        if kind == "segment":
            return segment_decisions(rows, cands, s)
        return timeline_decisions(rows, cands, s, centered=kind == "centered")

    results = {}
    for kind in ("segment", "centered", "trailing"):
        if not all(data[(kind, d)] for d in TRAIN + TEST):
            continue
        space, start = (
            (SEGMENT_SPACE, SEGMENT_START)
            if kind == "segment"
            else (TIMELINE_SPACE, TIMELINE_START)
        )
        best, train = tune(lambda s: mean(accuracy(run(kind, d, s)) for d in TRAIN), space, start)
        per = {d: accuracy(run(kind, d, best)) for d in TRAIN + TEST}
        ns = run(kind, "northstar", best)
        results[kind] = (best, train, per, switch_error(ns))

    print("## Picked on the FLEURS mixes, northstar held out\n")
    print(
        "| mode | setting | FLEURS mean | "
        + " | ".join(TRAIN)
        + " | northstar | northstar switch error |"
    )
    print("|---|---|---|" + "---|" * len(TRAIN) + "---|---|")
    for kind, (best, train, per, sw) in results.items():
        cols = " | ".join(f"{per[d]:.3f}" for d in TRAIN)
        print(
            f"| {kind} | {json.dumps(best)} | {train:.3f} | {cols} | {per['northstar']:.3f} | {sw:.1%} |"
        )

    for kind in ("centered", "trailing"):
        if kind not in results:
            continue
        best = results[kind][0]
        print(f"\n## Ablation from the picked {kind} setting (one change at a time)\n")
        print("| change | FLEURS mean | northstar | northstar switch error |\n|---|---|---|---|")
        variants = {
            "picked": {},
            "read=mean": {"read": "mean"},
            "p_gap off": {"p_gap": None},
            "whisper only": {"scorers": ("whisper-token",)},
            "windows 1,2,4": {"windows": ("1.0", "2.0", "4.0")},
        }
        for name, change in variants.items():
            s = {**best, **change}
            fl = mean(accuracy(run(kind, d, s)) for d in TRAIN)
            ns = run(kind, "northstar", s)
            print(f"| {name} | {fl:.3f} | {accuracy(ns):.3f} | {switch_error(ns):.1%} |")
        other = "trailing" if kind == "centered" else "centered"
        if other in results:
            s = best
            fl = mean(accuracy(run(other, d, s)) for d in TRAIN)
            ns = run(other, "northstar", s)
            print(
                f"| same setting, {other} windows | {fl:.3f} | {accuracy(ns):.3f} | {switch_error(ns):.1%} |"
            )
    json.dump(
        {
            k: {"setting": v[0], "fleurs": v[1], "per": v[2], "northstar_switch_error": v[3]}
            for k, v in results.items()
        },
        open(os.path.join(eval_dir, "grid2.json"), "w"),
        default=list,
    )


if __name__ == "__main__":
    main(sys.argv[1])
