"""A variant run next to its base run, on the same dataset.

    uv run --project bench python experiments/upgrade-260930/compare.py \
        --dataset fleurs_en-ko_worklog \
        --base asr.qwen-seg-v2-en+mt.qwen3.5-4b_off \
        --variant asr.qwen-seg-v2-en+mt.qwen3.5-4b_fragment [--items en_1845,en_1946]

Prints, from `bench/runs/<dataset>/<pipeline>/`:
  1. every summary metric of both runs and the difference,
  2. how often each log tag / DROP rule fired in each run (events.jsonl),
  3. the items whose transcript changed, with both transcripts (or only --items).

A parity check passes when section 3 is empty and section 1 differs only in wall-clock
metrics (`*_ca_*`, realtime, wall).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "bench" / "runs"
WALL_CLOCK = ("_ca", "wall", "compute", "realtime", "fsl")


def load(dataset: str, pipeline: str) -> tuple[dict, dict, Counter]:
    run = RUNS / dataset / pipeline
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    items, tags = {}, Counter()
    for line in (run / "items.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        items[row["id"]] = row
    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("type") != "log" or not event.get("tag"):
            continue
        tag = event["tag"]
        if tag == "DROP":
            rule = event.get("msg", "").split("rule=", 1)[-1].split(" ", 1)[0]
            tag = f"DROP {rule}"
        elif tag == "DEDUP-SKIP":
            rule = event.get("msg", "").split("rule=", 1)[-1].split(" ", 1)[0]
            tag = f"DEDUP-SKIP {rule}"
        tags[tag] += 1
    return summary, items, tags


def flat(metrics: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)):
            out[f"{prefix}{key}"] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--items", default="")
    args = parser.parse_args()

    base, base_items, base_tags = load(args.dataset, args.base)
    variant, variant_items, variant_tags = load(args.dataset, args.variant)

    print(f"# {args.dataset}: {args.base} -> {args.variant}")
    print(f"status {base['status']} -> {variant['status']}")
    a, b = flat(base["metrics"]), flat(variant["metrics"])
    print("\n## metrics (base, variant, variant - base)")
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        delta = "" if x is None or y is None else f"{y - x:+.4f}"
        mark = "  (wall clock)" if any(w in key for w in WALL_CLOCK) else ""
        print(f"  {key:34} {x!s:>12} {y!s:>12} {delta:>10}{mark}")

    print("\n## log tags (base, variant)")
    for tag in sorted(set(base_tags) | set(variant_tags)):
        if base_tags[tag] != variant_tags[tag]:
            print(f"  {tag:40} {base_tags[tag]:>6} {variant_tags[tag]:>6}")

    wanted = [i for i in args.items.split(",") if i]
    print("\n## transcripts that changed" + (f" (limited to {wanted})" if wanted else ""))
    changed = 0
    for item_id in wanted or sorted(set(base_items) & set(variant_items)):
        x = base_items.get(item_id, {}).get("transcription_output", "")
        y = variant_items.get(item_id, {}).get("transcription_output", "")
        if x != y or wanted:
            changed += x != y
            print(f"- {item_id}  wer {base_items.get(item_id, {}).get('wer')} -> "
                  f"{variant_items.get(item_id, {}).get('wer')}")
            print(f"    base:    {x}")
            print(f"    variant: {y}")
    print(f"\n{changed} item(s) changed of {len(set(base_items) & set(variant_items))}")


if __name__ == "__main__":
    main()
