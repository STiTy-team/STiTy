"""Turns part of a zero bench run into a scenario, and finds the parts worth turning.

The run must use a zero pipeline with `record_decodes: true`, so its events.jsonl has one
"decode" event per model call. A scenario points at the stretch of the item's audio; the run's
log lines for that stretch become its baseline (fixtures/<name>.jsonl, kept local).

    # where did the model loop or a slot grow long?
    python -m core.tests.zero_cut scan bench/runs/<dataset>/<pipeline>

    # cut item <id> from 120 s to 150 s into core/tests/zero_scenarios/<name>.yml
    python -m core.tests.zero_cut cut bench/runs/<dataset>/<pipeline> <id> <name> --from 120 --to 150

A cut starts right after a VAD flush (the slot is empty there) and ends at a VAD flush, so
the replay needs nothing from before it. Without --from it starts at the item's beginning;
without --to it runs to the end of the stream.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from core.tests.zero_replay import HERE, SR, longest_loop, spoken

CHUNK = SR * 200 // 1000  # bench feeds 200 ms pieces (bench/__main__.py CHUNK_SIZE_MS)


def read_items(run_dir: Path) -> dict[str, list[dict]]:
    items = defaultdict(list)
    with (run_dir / "events.jsonl").open() as f:
        for line in f:
            event = json.loads(line)
            if event["type"] == "run_open":
                items["__run__"] = [event]
            elif event.get("item") and event["type"] in ("chunk", "speech", "decode", "transcribed"):
                items[event["item"]].append(event)
    return items


def timeline(events: list[dict]) -> tuple[list[tuple[int, bool]], list[dict], list[dict]]:
    """The feed steps `(end sample, flushed)`, the decodes and the commits of one item."""
    chunks = [e for e in events if e["type"] == "chunk"]
    decodes = [e for e in events if e["type"] == "decode"]
    steps, flushed = [], False
    for event in events:
        if event["type"] == "speech":
            flushed = True
        elif event["type"] == "chunk":
            steps.append([CHUNK * (len(steps) + 1), flushed])
            flushed = False
    if steps:
        # Only the last piece can be short; a decode that reached it knows its exact end.
        last_full = CHUNK * (len(steps) - 1)
        ends = [d["slot_start"] + d["slot_samples"] for d in decodes]
        tail = [end for end in ends if end > last_full]
        steps[-1][0] = max(tail) if tail else round(chunks[-1]["audio"] * SR)
    commits = [
        {"original": e["original"], "language": e["language"], "commit_reason": e["commit_reason"], "audio": e.get("audio")}
        for e in events
        if e["type"] == "transcribed"
    ]
    return [tuple(s) for s in steps], decodes, commits


def scan(run_dir: Path) -> None:
    for item, events in read_items(run_dir).items():
        if item == "__run__":
            continue
        steps, decodes, commits = timeline(events)
        if not decodes:
            print(f"{item}: no decode events (record_decodes off?)")
            continue
        longest = max(decodes, key=lambda d: d["slot_samples"])
        reasons = defaultdict(int)
        for c in commits:
            reasons[c["commit_reason"]] += 1
        silent = sum(1 for d in decodes if not spoken(d["committed"] + d["output"]))
        print(
            f"\n{item}: {steps[-1][0] / SR:.1f}s, {len(decodes)} decodes ({silent} silent), "
            f"commits {dict(reasons)}, longest slot {longest['slot_samples'] / SR:.1f}s "
            f"at {longest['slot_start'] / SR:.1f}s"
        )
        for c in commits:
            times, phrase = longest_loop(c["original"])
            if times >= 3:
                print(f"  loop x{times} {phrase!r} at {c['audio']}s ({c['commit_reason']}): {c['original'][:120]}")


def cut(run_dir: Path, item: str, name: str, start: float | None, stop: float | None, what: str, why: str) -> None:
    items = read_items(run_dir)
    events = items.get(item)
    if not events:
        raise SystemExit(f"no events for item {item!r} in {run_dir}")
    steps, decodes, commits = timeline(events)
    if not decodes:
        raise SystemExit("this run has no decode events; run it with record_decodes: true")

    first = 0
    if start is not None:
        before = [i for i, (end, flushed) in enumerate(steps) if flushed and end <= start * SR]
        if not before:
            raise SystemExit(f"no VAD flush at or before {start}s to start from")
        first = before[-1] + 1
    offset = steps[first - 1][0] if first else 0
    last = len(steps) - 1
    if stop is not None:
        after = [i for i, (end, flushed) in enumerate(steps) if flushed and end >= stop * SR and i >= first]
        if not after:
            raise SystemExit(f"no VAD flush at or after {stop}s to end at")
        last = after[0]
    end = steps[last][0]

    # The run's own config names: "<dataset>@vN/<pipeline>" -> configs/... files of this checkout.
    run_open = items["__run__"][0]
    found_in = (run_open.get("source") or {}).get("commit", "?")[:8]
    dataset, _, pipeline = run_open["name"].partition("/")
    dataset, pipeline = dataset.split("@")[0], pipeline.split("@")[0]

    kept = [d for d in decodes if offset <= d["slot_start"] and d["slot_start"] + d["slot_samples"] <= end]
    inside = [c for c in commits if c["audio"] is not None and offset / SR < c["audio"] <= end / SR]
    fixture = HERE / "fixtures" / f"{name}.jsonl"
    fixture.parent.mkdir(exist_ok=True)
    with fixture.open("w") as f:
        f.write(json.dumps({"kind": "meta", "run": str(run_dir), "item": item, "start": offset, "end": end}) + "\n")
        for d in kept:
            row = {k: d[k] for k in ("slot_start", "slot_samples", "audio_sha1", "committed", "output")}
            f.write(json.dumps({"kind": "decode", **row}, ensure_ascii=False) + "\n")
        for c in inside:
            f.write(json.dumps({"kind": "commit", **c}, ensure_ascii=False) + "\n")

    scenario = HERE / f"{name}.yml"
    if scenario.exists():
        print(f"{scenario}: kept as it is; baseline rewritten ({len(kept)} decodes, {len(inside)} commits)")
        return
    scenario.write_text(
        f"what: {json.dumps(what or f'{item} {offset / SR:.1f}s-{end / SR:.1f}s', ensure_ascii=False)}\n"
        f"why: {json.dumps(why or '(왜 잘라 왔는지 적는다)', ensure_ascii=False)}\n"
        f"found_in: {found_in}\n"
        f"introduced_in: {{\"commit\": \"?\", \"by\": \"(원인이 들어온 커밋과 무엇 때문인지 적는다)\"}}\n"
        f"pipeline: configs/pipelines/{pipeline}.yml\n"
        f"dataset: configs/datasets/{dataset}.yml\n"
        f"item: {item}\n"
        f"span_sec: [{offset / SR:.1f}, {end / SR:.1f}]\n"
        f"span_samples: [{offset}, {end}]\n"
        f"run: {run_dir}\n"
        f"baseline: fixtures/{name}.jsonl\n"
        f"checks: []\n"
        f"known_issues: []\n"
    )
    print(f"{scenario}: {len(kept)} decodes, {len(inside)} commits, {offset / SR:.1f}s-{end / SR:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("scan")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("cut")
    p.add_argument("run_dir", type=Path)
    p.add_argument("item")
    p.add_argument("name")
    p.add_argument("--from", dest="start", type=float)
    p.add_argument("--to", dest="stop", type=float)
    p.add_argument("--what", default="")
    p.add_argument("--why", default="")
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.run_dir)
    else:
        cut(args.run_dir, args.item, args.name, args.start, args.stop, args.what, args.why)


if __name__ == "__main__":
    sys.exit(main())
