"""Compare a langid-reset run against its base run: lost and gained reference words, resets with
no real language switch nearby, and commits written in the other language's script.

- lost / gained words: reference words matched (difflib over normalised words) by one run's
  transcript and not by the other's.
- false resets: `SLOT-RESET cause=lid*` log lines with no reference language switch in
  [t - LOOKBACK, t + AHEAD] (log seconds = session audio seconds; sessions in items order).
- wrong-script commits: an English commit (route truth) holding Hangul, or a Korean one holding
  more Latin letters than Hangul.

usage: uv run --project bench python core/research/lid_module/reset_metrics.py \
         --base bench/runs/northstar_ko+en-ko@v4/common_audiolid-timeline \
         --new bench/runs/northstar_ko+en-ko@v4/common_audiolid-timeline-lidreset-boundary \
         --log logs/audio-lid/lidreset3.boundary.northstar_ko+en-ko.log
"""

import argparse
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from bench.metrics.asr import _routes  # noqa: E402

LOOKBACK, AHEAD = 5.0, 0.5
RESET_RE = re.compile(r"^INFO:\s+([\d.]+)s SLOT-RESET cause=(lid[\w-]*)(.*)$")


def norm_words(text: str) -> list[str]:
    return [w for w in (re.sub(r"[^\w]+", "", t.lower()) for t in text.split()) if w]


def matched(ref: list[str], hyp: list[str]) -> set[int]:
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return {i for a, _, n in sm.get_matching_blocks() for i in range(a, a + n)}


def rows(run: str) -> list[dict]:
    return [json.loads(line) for line in open(os.path.join(run, "items.jsonl"), encoding="utf-8")]


def commits(row: dict) -> str:
    return " ".join(r["original"] for r in row["records"] if r["type"] == "transcribed")


def wrong_script(row: dict) -> list[str]:
    out = []
    trans = [r for r in row["records"] if r["type"] == "translated"]
    for f, (truth, _) in zip(trans, _routes(row)):
        t = f["original"]
        h, lat = len(re.findall("[가-힣]", t)), len(re.findall("[A-Za-z]", t))
        if (truth == "en" and h) or (truth == "ko" and lat > h):
            out.append(t)
    return out


def switches(row: dict) -> list[float]:
    out, cur = [], None
    for w in row["reference_alignment"]:
        lang = "ko" if re.search("[가-힣]", w["word"]) else "en" if re.search("[A-Za-z]", w["word"]) else ""
        if lang and lang != cur:
            if cur:
                out.append(w["start"])
            cur = lang
    return out


def resets(log_path: str, n_sessions: int) -> list[list[tuple[float, str, str]]]:
    per, last = [[]], 0.0
    started = False
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = re.match(r"^INFO:\s+([\d.]+)s ", line)
        if not m:
            continue
        t = float(m.group(1))
        if started and t + 5 < last:
            per.append([])
        last, started = t, True
        r = RESET_RE.match(line.rstrip())
        if r:
            # Prefer the audio time the reset line carries; log seconds drift behind the audio.
            at = re.search(r"audio=([\d.]+)", r.group(3))
            per[-1].append((float(at.group(1)) if at else t, r.group(2), r.group(3)[:160]))
    return (per + [[]] * n_sessions)[:n_sessions]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    base, new = rows(a.base), rows(a.new)
    tot = {"lost": 0, "gained": 0, "false": 0, "resets": 0, "ws_base": 0, "ws_new": 0}
    for b, n, rs in zip(base, new, resets(a.log, len(new))):
        ref = norm_words(b["reference"])
        mb, mn = matched(ref, norm_words(commits(b))), matched(ref, norm_words(commits(n)))
        lost, gained = sorted(mb - mn), sorted(mn - mb)
        sw = switches(n)
        false = [r for r in rs if not any(r[0] - LOOKBACK <= s <= r[0] + AHEAD for s in sw)]
        wb, wn = wrong_script(b), wrong_script(n)
        print(f"== {n['id']}: lost {len(lost)}, gained {len(gained)}, resets {len(rs)} "
              f"(false {len(false)}), wrong-script {len(wb)} -> {len(wn)}")
        if lost:
            print("   lost  :", " ".join(ref[i] for i in lost))
        for r in false:
            print(f"   false reset {r[0]:.1f}s {r[1]}{r[2]}")
        for t in sorted(set(wb) - set(wn)):
            print("   fixed :", t)
        for t in sorted(set(wn) - set(wb)):
            print("   new ws:", t)
        tot["lost"] += len(lost)
        tot["gained"] += len(gained)
        tot["false"] += len(false)
        tot["resets"] += len(rs)
        tot["ws_base"] += len(wb)
        tot["ws_new"] += len(wn)
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
