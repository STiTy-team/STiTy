"""Merge the first labels, their judge verdicts, the retry labels and their verdicts into one label file.

A sid lands in exactly one place: the first label if judge 1 passed it (or could not judge it), else the retry
label if judge 2 passed it, else the rescue list (why = "judge"); a labeler FAIL/ERROR with no label at all is a
rescue too (why = "fail"). Rescued sids become wait labels in schedule_stats.py.

usage: PYTHONPATH=. .venv/bin/python core/research/zprobe/labels/compose_judged.py \
         --s1 schedule.jsonl --j1 schedule.judge1.jsonl --s2 schedule.retry.jsonl --j2 schedule.retry.judge2.jsonl \
         --fail1 schedule.jsonl.fail.jsonl --fail2 schedule.retry.jsonl.fail.jsonl \
         --out schedule.judged.jsonl --out-rescue rescue.jsonl"""
import argparse
import json


def _by_sid(rows):
    return {r["sid"]: r for r in rows}


def compose(s1, j1, s2, j2, fail_sids=()):
    """-> (kept rows with judge_attempts, rescue [{sid, why}]). Rows are dicts; j rows carry pass/unjudged."""
    S1, J1, S2, J2 = _by_sid(s1), _by_sid(j1), _by_sid(s2), _by_sid(j2)
    kept, rescue, placed = [], [], set()
    for sid, r in S1.items():
        v1 = J1.get(sid)
        if v1 is None or v1.get("pass") or v1.get("unjudged"):
            kept.append({**r, "judge_attempts": 1})
        elif sid in S2 and J2.get(sid, {}).get("pass"):
            kept.append({**S2[sid], "judge_attempts": 2})
        else:
            rescue.append({"sid": sid, "why": "judge"})
        placed.add(sid)
    for sid in fail_sids:
        if sid not in placed:
            rescue.append({"sid": sid, "why": "fail"})
            placed.add(sid)
    return kept, rescue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--j1", required=True)
    ap.add_argument("--s2")
    ap.add_argument("--j2")
    ap.add_argument("--fail1")
    ap.add_argument("--fail2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-rescue", required=True)
    a = ap.parse_args()
    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p else []
    fails = [r["sid"] for r in load(a.fail1) + load(a.fail2)]
    kept, rescue = compose(load(a.s1), load(a.j1), load(a.s2), load(a.j2), fails)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out_rescue, "w", encoding="utf-8") as f:
        for r in rescue:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n2 = sum(r["judge_attempts"] == 2 for r in kept)
    print(f"COMPOSE_DONE kept={len(kept)} (retry {n2}) rescue={len(rescue)} "
          f"(judge {sum(r['why'] == 'judge' for r in rescue)}, fail {sum(r['why'] == 'fail' for r in rescue)})")


if __name__ == "__main__":
    main()
