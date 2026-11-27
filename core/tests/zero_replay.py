"""Replays zero scenarios: each scenario's stretch of speech goes through the real pipeline.

From the repo root, on a machine with a GPU and the datasets:

    python -m core.tests.zero_replay                         # every scenario
    python -m core.tests.zero_replay limit-cut-mid-switch    # by name
    python -m core.tests.zero_replay --pipeline configs/pipelines/<other>.yml   # try another pipeline

A scenario (core/tests/zero_scenarios/<name>.yml) is a problem seen in a bench run: what went
wrong, the stretch of the item's audio where it happened, and that run's log lines for the
stretch (fixtures/<name>.jsonl, the baseline). The replay feeds the stretch to the pipeline in
200 ms pieces, as bench does, records every event, and shows the commits next to the baseline.

- `checks` must hold now. A failure exits non-zero.
- `known_issues` are the reported problems. They fail until fixed; then the run says FIXED and
  the entry moves to `checks`.

Each replay's events go to zero_scenarios/replays/<name>/<time>.jsonl, one line per scenario to
zero_scenarios/results.jsonl, and the latest state of every scenario to zero_scenarios/STATUS.md.
All of these stay local.
"""

import argparse
import asyncio
import datetime
import difflib
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from core.components.transcription.zero_qwen3 import ASR_TEXT, SEG
from core.utils import audio as audio_mod
from core.utils import stream

HERE = Path(__file__).parent / "zero_scenarios"
RESULTS = HERE / "results.jsonl"
STATUS = HERE / "STATUS.md"
SR = audio_mod.SAMPLING_RATE
# How alike two commits' text must be to count as the same commit decoded a little differently.
GPU_TEXT_SIMILARITY = 0.8
CHUNK_MS = 200  # bench/__main__.py CHUNK_SIZE_MS
TRAILING_SILENCE_MS = 4000  # bench/__main__.py


def spoken(text: str) -> str:
    """The words in a raw decode, without language headers and <SEG>."""
    body = re.sub(r"language \S*" + re.escape(ASR_TEXT), " ", text).replace(SEG, " ")
    return " ".join(body.split())


def longest_loop(text: str, longest: int = 6) -> tuple[int, str]:
    """How many times in a row one phrase (1..`longest` words) repeats, and the phrase."""
    words, best = text.split(), (1, "")
    for n in range(1, longest + 1):
        for i in range(len(words) - n):
            times = 1
            while words[i + times * n : i + (times + 1) * n] == words[i : i + n]:
                times += 1
            if times > best[0]:
                best = (times, " ".join(words[i : i + n]))
    return best


@dataclass
class Replay:
    commits: list[dict]  # transcribed events; "audio" is seconds into the item
    decodes: list[dict]
    baseline: list[dict]

    def reasons(self) -> Counter:
        return Counter(c["commit_reason"] for c in self.commits)

    def longest_slot_sec(self) -> float:
        return max((d["slot_samples"] / SR for d in self.decodes), default=0.0)

    def metrics(self) -> dict:
        return {
            "commits": len(self.commits),
            "decodes": len(self.decodes),
            "silent_decodes": sum(1 for d in self.decodes if not spoken(d["committed"] + d["output"])),
            "longest_slot_sec": round(self.longest_slot_sec(), 2),
            "reasons": dict(sorted(self.reasons().items())),
            "same_as_baseline": self.same_as_baseline(),
            "behavior": self.behavior()[0],
        }

    def same_as_baseline(self) -> bool:
        return [c["original"] for c in self.commits] == [c["original"] for c in self.baseline]

    def behavior(self) -> tuple[str, list[str]]:
        """"same", "text-only" (every commit where it was, a few characters off, as vLLM's
        decoding drifts between runs) or "different", with the commits that differ."""
        shape = lambda c: (c["commit_reason"], c["language"], round(c["audio"], 1))
        if [shape(c) for c in self.commits] != [shape(c) for c in self.baseline]:
            changed = [
                line for line in difflib.unified_diff(
                    [f"{c['audio']:.1f}s {c['commit_reason']} [{c['language']}] {c['original']}" for c in self.baseline],
                    [f"{c['audio']:.1f}s {c['commit_reason']} [{c['language']}] {c['original']}" for c in self.commits],
                    lineterm="", n=0)
                if line[:1] in "+-" and line[:3] not in ("+++", "---")
            ]
            return "different", changed
        off = [
            f"{a['audio']:.1f}s {a['original']!r} -> {b['original']!r}"
            for a, b in zip(self.baseline, self.commits)
            if a["original"] != b["original"]
        ]
        if not off:
            return "same", []
        alike = all(
            difflib.SequenceMatcher(a=a["original"], b=b["original"]).ratio() >= GPU_TEXT_SIMILARITY
            for a, b in zip(self.baseline, self.commits)
        )
        return ("text-only" if alike else "different"), off


def bare(text: str) -> str:
    """Text as bench scores it, then without spaces, punctuation and case.

    The same chain as WER scoring with r2 on (bench/metrics/transcription.py): r2norm (fillers,
    spelling variants, dialect endings, 이차 -> 2 차), Korean number forms, then the basic
    normaliser. "이차" and "2차", "This was" and "this was", "JMT" and "존맛탱" all match.
    """
    from bench.metrics import r2norm
    from bench.metrics.common import basic
    from bench.metrics.numbers import normalize_numbers

    return re.sub(r"[\W_]", "", basic(normalize_numbers(r2norm.apply(text, r2norm.parts("all")))).lower())


def said(r, text) -> bool:
    """Whether a commit holds `text`; a list means any one of its alternatives."""
    options = text if isinstance(text, list) else [text]
    return any(bare(t) in bare(c["original"]) for t in options for c in r.commits)


# Each check takes the replay and its parameters from the YAML and says whether it holds.
# Text is compared bare (see `bare`); in `texts`, a list stands for alternatives.
CHECKS = {
    "must-commit": lambda r, texts: all(said(r, t) for t in texts),
    "must-commit-any": lambda r, texts: any(said(r, t) for t in texts),
    "never-commits": lambda r, texts: not any(said(r, t) for t in texts),
    "allowed-languages": lambda r, codes: all(c["language"] in codes for c in r.commits),
    "language-of": lambda r, text, code: all(c["language"] == code for c in r.commits if bare(text) in bare(c["original"])),
    "longest-slot-under": lambda r, sec: r.longest_slot_sec() <= sec,
    "no-commit-reason": lambda r, reason: reason not in r.reasons(),
    "no-repeat-loop": lambda r, max: all(longest_loop(c["original"])[0] <= max for c in r.commits),
    "same-as-baseline": lambda r: r.same_as_baseline(),
    "behaves-as-baseline": lambda r: r.behavior()[0] != "different",
}


def parse_check(entry) -> tuple[str, dict]:
    name, params = (entry, {}) if isinstance(entry, str) else next(iter(entry.items()))
    if name not in CHECKS:
        raise SystemExit(f"unknown check {name!r}; known: {sorted(CHECKS)}")
    return name, params or {}


def label(name: str, params: dict) -> str:
    return name + ("" if not params else " " + json.dumps(params, ensure_ascii=False))


def applies(sc: dict, settings: dict) -> list[str]:
    """The `applies_when` conditions the pipeline's transcription settings do not meet.

    true: the setting is on; "set": it is given (any value); anything else: equal to it.
    """
    missing = []
    for key, want in (sc.get("applies_when") or {}).items():
        have = settings.get(key)
        ok = bool(have) if want is True else have not in (None, False) if want == "set" else have == want
        if not ok:
            missing.append(f"{key}={json.dumps(want)}")
    return missing


def read_baseline(path: Path) -> tuple[dict, list[dict]]:
    meta, commits = None, []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "meta":
            meta = row
        elif row["kind"] == "commit":
            commits.append(row)
    return meta, commits


class Bench:
    """One pipeline and one dataset, loaded once for every scenario that uses them."""

    def __init__(self, pipeline: str, dataset: str) -> None:
        from bench import augment, shared_configs
        from bench import dataset as datasets
        from bench.settings import BenchSettings
        from core.pipeline import build

        settings = BenchSettings.load()
        self.cfg = shared_configs.load_for_run(pipeline, dataset, None, queued_job=False)
        self.dataset = datasets.load(self.cfg.dataset, settings.stity_data_root)
        self.augmenter = augment.build(self.cfg.augment, data_root=settings.stity_data_root)
        self.items = {item.id: item for item in self.dataset.items}
        self.pipeline = build(self.cfg)

    async def load(self) -> None:
        await self.pipeline.load()

    def audio(self, item_id: str) -> bytes:
        """The item's stream exactly as bench feeds it: augmented, then 4 s of silence."""
        from bench import augment

        item = self.items[item_id]
        samples = audio_mod.load_window(item.audio, offset=item.offset, duration=item.duration_sec)
        samples, _ = self.augmenter.apply(samples, key=augment.item_key(self.cfg.dataset.name, item.id))
        return audio_mod.to_pcm_bytes(samples) + audio_mod.silence_bytes(TRAILING_SILENCE_MS)

    async def replay(self, item_id: str, start: int, end: int, events: Path) -> tuple[list, list]:
        """Feeds samples start..end of the item and returns its commits and decodes.

        VAD remembers what it heard, so the audio before the stretch goes through VAD alone
        first; its speech ends then fall where they fell in the full run.
        """
        whole = self.audio(item_id)
        before, pcm = whole[: start * 2], whole[start * 2 : end * 2]
        step = SR * CHUNK_MS // 1000 * 2
        stream.attach(events)
        stream.bind(item=item_id, session=item_id)
        records = []
        try:
            self.pipeline.start(
                languages=self.dataset.languages, target_lang=self.cfg.target, own_lang=self.cfg.lang
            )
            vad = self.pipeline.part("vad")
            if vad is not None:
                for at in range(0, len(before), step):
                    vad.detect(before[at : at + step])
            heard = start / SR
            for at in range(0, len(pcm), step):
                chunk = pcm[at : at + step]
                heard += len(chunk) / 2 / SR
                stream.audio_position(heard)
                records += [stream.record(m.type, **asdict(m)) for m in await self.pipeline.listen(chunk)]
                stream.record("chunk", silence=False)
            records += [stream.record(m.type, **asdict(m)) for m in await self.pipeline.finish()]
        finally:
            stream.audio_position(None)
            stream.close()
        decodes = [json.loads(line) for line in events.read_text().splitlines()]
        decodes = [e for e in decodes if e["type"] == "decode"]
        commits = [r for r in records if r["type"] == "transcribed"]
        return commits, decodes


def describe(sc: dict, replay: Replay) -> str:
    lines = []
    for side, commits in (("before", replay.baseline), ("now", replay.commits)):
        lines.append(f"  {side}:")
        lines += [f"    {c['audio']:7.1f}s {c['commit_reason']:>6} [{c['language']}] {c['original']}" for c in commits]
    lines.append("    " + json.dumps(replay.metrics(), ensure_ascii=False))
    return "\n".join(lines)


def git_state() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", "core"))


def write_status(latest: dict) -> None:
    lines = [
        "# zero 시나리오 상태",
        "",
        "`python -m core.tests.zero_replay` 가 실행마다 다시 쓴다. 시나리오마다 마지막 replay 결과.",
        "",
        "| 시나리오 | 언제 · 커밋 · 파이프라인 | 반드시 맞을 것 | 보고된 문제 | 수치 |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(latest):
        row = latest[name]
        failed = [k for k, ok in row["checks"].items() if not ok]
        known = [k for k, ok in row["known_issues"].items() if not ok]
        fixed = [k for k, ok in row["known_issues"].items() if ok]
        checks = "통과" if not failed else "실패: " + ", ".join(failed)
        issues = ", ".join(known) or "-"
        if fixed:
            issues += " / FIXED: " + ", ".join(fixed)
        m = row["metrics"]
        behavior = {"same": "같음", "text-only": "글자만 다름(GPU)", "different": "다른 동작"}[m["behavior"]]
        numbers = (f"커밋 {m['commits']} · 디코딩 {m['decodes']} · 최장 슬롯 {m['longest_slot_sec']}s · "
                   f"기준선과 {behavior}")
        commit = row["commit"] + ("+수정" if row["dirty"] else "")
        lines.append(f"| `{name}` | {row['at']} · {commit} · {row['pipeline']} | {checks} | {issues} | {numbers} |")
    STATUS.write_text("\n".join(lines) + "\n")


def in_history(sc: dict) -> bool:
    """Whether the commit that brought the scenario's cause is in this checkout's history (a
    branch rebuilt without it has no such problem to look for). guard and unknown pass."""
    commit = (sc.get("introduced_in") or {}).get("commit", "")
    if not commit or commit in ("guard", "?"):
        return True
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True).returncode == 0


def _not_applicable(name: str, sc: dict, settings: dict) -> bool:
    missing = applies(sc, settings)
    if missing:
        print(f"[{name}] not applicable to this pipeline (needs {', '.join(missing)})")
    elif not in_history(sc):
        print(f"[{name}] not applicable to this checkout ({sc['introduced_in']['commit']} is not in its history)")
        return True
    return bool(missing)


async def main(names: list[str], pipeline_override: str | None, gate: bool) -> int:
    paths = sorted(HERE.glob("*.yml"))
    if names:
        paths = [p for p in paths if p.stem in names]
    scenarios = [(p.stem, yaml.safe_load(p.read_text())) for p in paths]
    commit, dirty = git_state()
    at = datetime.datetime.now().isoformat(timespec="minutes")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    groups: dict[tuple[str, str], list] = {}
    for name, sc in scenarios:
        if not (HERE / sc["baseline"]).exists():
            print(f"[{name}] skipped: {sc['baseline']} is not on this machine (baselines stay local)")
            continue
        groups.setdefault((pipeline_override or sc["pipeline"], sc["dataset"]), []).append((name, sc))

    failed = fixed = 0
    rows = []
    for (pipeline, dataset), members in groups.items():
        print(f"\n### {pipeline} on {dataset}: loading")
        bench = Bench(pipeline, dataset)
        settings = bench.cfg.stity.resolved["parts"]["transcription"]["kwargs"]
        members = [(name, sc) for name, sc in members if not _not_applicable(name, sc, settings)]
        if not members:
            continue
        await bench.load()
        for name, sc in members:
            meta, baseline = read_baseline(HERE / sc["baseline"])
            events = HERE / "replays" / name / f"{stamp}.jsonl"
            events.parent.mkdir(parents=True, exist_ok=True)
            commits, decodes = await bench.replay(sc["item"], meta["start"], meta["end"], events)
            replay = Replay(commits, decodes, baseline)

            print(f"\n[{name}] {sc['what']}")
            introduced = sc.get("introduced_in") or {}
            print(f"  found in {sc.get('found_in', '?')}, introduced in {introduced.get('commit', '?')}: {introduced.get('by', '')}")
            print(describe(sc, replay))
            verdict, differences = replay.behavior()
            print(f"  behavior vs baseline: {verdict}")
            for line in differences:
                print(f"    {line}")
            checks, issues = {}, {}
            if gate:
                checks["behaves-as-baseline"] = verdict != "different"
            for entry in sc.get("checks", []):
                check, params = parse_check(entry)
                checks[label(check, params)] = CHECKS[check](replay, **params)
            for entry in sc.get("known_issues", []):
                check, params = parse_check(entry)
                issues[label(check, params)] = CHECKS[check](replay, **params)
            for check, ok in checks.items():
                failed += not ok
                print(f"  {'ok  ' if ok else 'FAIL'} {check}")
            for check, ok in issues.items():
                fixed += ok
                print(f"  {'FIXED' if ok else 'known'} {check}")
            rows.append({"at": at, "commit": commit, "dirty": dirty, "pipeline": Path(pipeline).stem,
                         "scenario": name, "events": str(events.relative_to(HERE)),
                         "checks": checks, "known_issues": issues, "metrics": replay.metrics()})
        await bench.pipeline.close()

    with RESULTS.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    latest = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            row = json.loads(line)
            if (HERE / f"{row['scenario']}.yml").exists():
                latest[row["scenario"]] = row
    write_status(latest)

    print(f"\n{failed} check(s) failed, {fixed} known issue(s) now pass  ->  {STATUS}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--pipeline", help="replay with this pipeline config instead of each scenario's own")
    parser.add_argument("--gate", action="store_true",
                        help="also fail when a replay behaves differently from its baseline (text-only drift passes)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.names, args.pipeline, args.gate)))
