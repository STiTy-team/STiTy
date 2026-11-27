"""Scenario tests for the zero transcriber's slot resets, without a GPU.

From the repo root: python -m core.tests.test_zero_qwen3 [name ...]

Scenarios are the YAML files in core/tests/zero_scenarios/ (how to write one is in the README
there). Each run appends one line per scenario to zero_scenarios/results.jsonl and rewrites
zero_scenarios/STATUS.md; both stay local.

- `checks` must hold now. A failure exits non-zero.
- `known_issues` are problems found and not fixed yet. They are expected to fail; when one
  starts passing the run says FIXED, and it should move to `checks`.

Every scenario also gets two checks of its own: the engine stayed in step (no decode the
script or the recording cannot answer) and a second run gives the same output.
"""

import asyncio
import datetime
import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from core.components.transcription.zero_qwen3 import ASR_TEXT, SEG
from core.tests.zero_fake import (
    SR,
    ReplayEngine,
    ScriptEngine,
    Word,
    even_steps,
    feed,
    sentence,
    transcriber,
)

HERE = Path(__file__).parent / "zero_scenarios"
RESULTS = HERE / "results.jsonl"
STATUS = HERE / "STATUS.md"


@dataclass
class Run:
    committed: list  # Transcribed, in order
    engine: object
    words: list[Word] | None = None  # script scenarios
    recorded: list[dict] | None = None  # replay scenarios: the commits of the bench run

    @property
    def written(self) -> list[str]:
        return [t.rstrip(".") for item in self.committed for t in item.original.split()]

    def lost(self) -> list[str]:
        said = [w.text.rstrip(".") for w in self.words or []]
        return [w for w in said if w not in self.written]

    def extra(self) -> list[str]:
        said = {w.text.rstrip(".") for w in self.words or []}
        return [w for w in self.written if w not in said]

    def repeated(self) -> list[str]:
        if self.words is None:
            return []
        return sorted({w for w in self.written if self.written.count(w) > 1})

    def reasons(self) -> list[str]:
        return [item.commit_reason for item in self.committed]

    def longest_slot_sec(self) -> float:
        return max(((d.window[1] - d.window[0]) / SR for d in self.engine.decodes), default=0.0)

    def silent_decodes(self) -> int:
        return sum(1 for d in self.engine.decodes if not spoken(d.committed + d.output))

    def misses(self) -> int:
        return sum(d.missed for d in self.engine.decodes)

    def metrics(self) -> dict:
        out = {
            "commits": len(self.committed),
            "decodes": len(self.engine.decodes),
            "misses": self.misses(),
            "silent_decodes": self.silent_decodes(),
            "longest_slot_sec": round(self.longest_slot_sec(), 2),
            "reasons": {r: self.reasons().count(r) for r in sorted(set(self.reasons()))},
        }
        if self.words is not None:
            out.update(lost=self.lost(), repeated=self.repeated(), extra=self.extra())
        return out


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


# Each check takes the run and its parameters from the YAML and says whether it holds.
CHECKS = {
    "every-word-once": lambda r: not r.lost() and not r.repeated(),
    "no-extra-words": lambda r: not r.extra(),
    "language-per-sentence": lambda r: all(sentence_lang(r, i) in ("", i.language) for i in r.committed),
    "allowed-languages": lambda r, codes: all(i.language in codes for i in r.committed),
    "must-commit": lambda r, texts: all(any(t in i.original for i in r.committed) for t in texts),
    "language-of": lambda r, text, code: all(i.language == code for i in r.committed if text in i.original),
    "first-commit-reason": lambda r, reason: r.reasons()[:1] == [reason],
    "last-commit-reason": lambda r, reason: r.reasons()[-1:] == [reason],
    "has-commit-reason": lambda r, reason: reason in r.reasons(),
    "commit-reason-count": lambda r, reason, min: r.reasons().count(reason) >= min,
    "slot-restarted": lambda r, after: any(d.window[0] >= after * SR for d in r.engine.decodes),
    "slot-never-restarted": lambda r: all(d.window[0] == r.engine.decodes[0].window[0] for d in r.engine.decodes),
    "longest-slot-under": lambda r, sec: r.longest_slot_sec() <= sec,
    "language": lambda r, code: all(i.language == code for i in r.committed),
    "no-empty-commit": lambda r: all(i.original.strip() for i in r.committed),
    "no-silent-decode": lambda r: r.silent_decodes() == 0,
    "matches-recording": lambda r: [i.original for i in r.committed] == [c["original"] for c in r.recorded],
    "never-commits": lambda r, texts: not any(t in i.original for t in texts for i in r.committed),
    "no-repeat-loop": lambda r, max: all(longest_loop(i.original)[0] <= max for i in r.committed),
}


def sentence_lang(run: Run, item) -> str:
    """The language most of a commit's words were spoken in ("" if none are known)."""
    langs = {w.text.rstrip("."): w.lang for w in run.words or []}
    spoken_in = [langs[t.rstrip(".")] for t in item.original.split() if t.rstrip(".") in langs]
    return max(set(spoken_in), key=spoken_in.count) if spoken_in else ""


def parse_check(entry) -> tuple[str, dict]:
    name, params = (entry, {}) if isinstance(entry, str) else next(iter(entry.items()))
    if name not in CHECKS:
        raise SystemExit(f"unknown check {name!r}; known: {sorted(CHECKS)}")
    return name, params or {}


def label(name: str, params: dict) -> str:
    return name + ("" if not params else " " + json.dumps(params, ensure_ascii=False))


def build_talk(entries: list[dict]) -> list[Word]:
    words = []
    for entry in entries:
        if "sentence" in entry:
            spec = dict(entry["sentence"])
            times, every = spec.pop("repeat", 1), spec.pop("every", 0.0)
            for k in range(times):
                words += sentence(**{**spec, "first": spec["first"] + k * spec["count"], "start": spec["start"] + k * every})
        else:
            words.append(Word(**entry["word"]))
    return words


def load_fixture(path: Path) -> tuple[dict, list[dict], list[dict]]:
    meta, decodes, commits = None, [], []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        kind = row.pop("kind")
        if kind == "meta":
            meta = row
        elif kind == "decode":
            decodes.append(row)
        else:
            commits.append(row)
    return meta, decodes, commits


async def play(sc: dict) -> Run:
    if sc["source"] == "replay":
        meta, decodes, commits = load_fixture(HERE / sc["fixture"])
        engine = ReplayEngine(decodes)
        asr = transcriber(engine, offset=meta["offset"])
        steps = [tuple(s) for s in meta["steps"]]
        out = await feed(asr, steps, start=meta["offset"], finish=meta["finish"])
        return Run(out, engine, recorded=commits)
    words = build_talk(sc["talk"])
    model = sc.get("model", {})
    engine = ScriptEngine(words, empty_at=model.get("empty_at", ()), guess=model.get("guess", False))
    steps = even_steps(sc["until"], vad_ends=sc.get("vad_ends", ()))
    out = await feed(transcriber(engine), steps, finish=False)
    return Run(out, engine, words=words)


def describe(run: Run) -> str:
    lines = [f"    {item.commit_reason:>6}: {item.original}" for item in run.committed[:40]]
    if len(run.committed) > 40:
        lines.append(f"    ... {len(run.committed) - 40} more")
    if run.words is not None:
        said = [w.text.rstrip(".") for w in run.words]
        diff = [line for line in difflib.ndiff(said, run.written) if line[:1] in "+-"]
        lines.append(f"    diff={diff}")
    lines.append("    " + json.dumps(run.metrics(), ensure_ascii=False))
    return "\n".join(lines)


def git_state() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", "core"))


def write_status(latest: dict) -> None:
    lines = [
        "# zero 시나리오 상태",
        "",
        "`python -m core.tests.test_zero_qwen3` 가 실행마다 다시 쓴다. 시나리오마다 마지막 결과.",
        "",
        "| 시나리오 | 언제 · 커밋 | 반드시 맞을 것 | 아직 안 고친 것 | 수치 |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(latest):
        row = latest[name]
        if row.get("stale"):
            lines.append(f"| `{name}` | {row['at']} · {row['commit']} | 녹화 밖 (GPU로 다시 녹화) | - | 녹화에 없음 {row['metrics']['misses']} |")
            continue
        failed = [k for k, ok in row["checks"].items() if not ok]
        known = [k for k, ok in row["known_issues"].items() if not ok]
        fixed = [k for k, ok in row["known_issues"].items() if ok]
        checks = "통과" if not failed else "실패: " + ", ".join(failed)
        issues = ", ".join(known) or "-"
        if fixed:
            issues += " / FIXED: " + ", ".join(fixed)
        m = row["metrics"]
        numbers = f"커밋 {m['commits']} · 디코딩 {m['decodes']} · 최장 슬롯 {m['longest_slot_sec']}s"
        if m.get("lost"):
            numbers += f" · 빠짐 {m['lost']}"
        if m["misses"]:
            numbers += f" · 녹화에 없음 {m['misses']}"
        commit = row["commit"] + ("+수정" if row["dirty"] else "")
        lines.append(f"| `{name}` | {row['at']} · {commit} | {checks} | {issues} | {numbers} |")
    STATUS.write_text("\n".join(lines) + "\n")


async def main(names: list[str]) -> int:
    paths = sorted(HERE.glob("*.yml"))
    if names:
        paths = [p for p in paths if p.stem in names]
    commit, dirty = git_state()
    at = datetime.datetime.now().isoformat(timespec="minutes")
    failed = fixed = 0
    rows = []
    for path in paths:
        sc = yaml.safe_load(path.read_text())
        if sc["source"] == "replay" and not (HERE / sc["fixture"]).exists():
            print(f"\n[{path.stem}] skipped: {sc['fixture']} is not on this machine (fixtures stay local)")
            continue
        run, again = await play(sc), await play(sc)
        same = [i.original for i in run.committed] == [i.original for i in again.committed]

        print(f"\n[{path.stem}] {sc['what']}")
        print(describe(run))
        if sc["source"] == "replay" and run.misses():
            # The logic asked for decodes the recording does not have: the replay no longer
            # says what the model would do. Re-record the span on a GPU.
            print(f"\n[{path.stem}] STALE: {run.misses()} of {len(run.engine.decodes)} decodes are not in the recording")
            rows.append({"at": at, "commit": commit, "dirty": dirty, "scenario": path.stem, "stale": True,
                         "checks": {}, "known_issues": {}, "metrics": run.metrics()})
            continue
        checks = {"engine stayed in step": run.misses() == 0, "same output on a second run": same}
        for entry in sc.get("checks", []):
            name, params = parse_check(entry)
            checks[label(name, params)] = CHECKS[name](run, **params)
        issues = {}
        for entry in sc.get("known_issues", []):
            name, params = parse_check(entry)
            issues[label(name, params)] = CHECKS[name](run, **params)
        for name, ok in checks.items():
            failed += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        for name, ok in issues.items():
            fixed += ok
            print(f"  {'FIXED' if ok else 'known'} {name}")
        rows.append(
            {"at": at, "commit": commit, "dirty": dirty, "scenario": path.stem,
             "checks": checks, "known_issues": issues, "metrics": run.metrics()}
        )

    with RESULTS.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    latest = {}
    for line in RESULTS.read_text().splitlines():
        row = json.loads(line)
        if (HERE / f"{row['scenario']}.yml").exists():
            latest[row["scenario"]] = row
    write_status(latest)

    print(f"\n{failed} check(s) failed, {fixed} known issue(s) now pass  ->  {STATUS}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
