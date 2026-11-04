"""Context-aware LLM judge for DialogueContext translations.

One judge call per unique (instance_id, hypothesis) pair: many conditions and models
produce the same string, and the judge never learns which model or condition produced
a candidate, so one verdict serves every row that carries that string.

The judge sees the whole earlier dialogue (Korean source turns with speaker labels and
personas, plus the gold English translations), the current turn, the reference, the
instance's `challenge_tags` and `checks`, and the candidate. It returns six 1-5 scores,
error labels, a pass/fail per check and a short rationale.

Cost tracking (see .claude/rules/cost-watch.md):
  - every API call appends usage, per-call cost and cumulative cost to judge_usage.jsonl
  - the cumulative cost resumes from that file, so --budget-usd covers all runs of this
    run dir; the judge stops launching calls once the budget would be crossed
  - finished verdicts are appended to judge_cache.jsonl, so a crash loses nothing

Runs in venv-tf5 (has openai + python-dotenv):

    venv-tf5/bin/python evaluation/DialogueContext/scripts/judge_context.py \
        --run-dir evaluation/DialogueContext/results/<run_id> [--model gpt-6-sol] \
        [--budget-usd 12] [--concurrency 4] [--limit N] [--show-prompt]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import scoring_common as common

JUDGE_VERSION = "dc-judge-2026-09-25.2"

# USD per 1M tokens: (input, cached input, output). 2026-09-25 OpenAI price list (DESIGN.md).
PRICES = {
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

DIMENSIONS = [
    "meaning_preservation",
    "no_hallucination",
    "contextual_correctness",
    "speaker_consistency",
    "register_consistency",
    "naturalness",
]
ERROR_LABELS = [
    "omission", "addition", "wrong_coreference", "wrong_gender", "wrong_register",
    "lexical_inconsistency", "unnatural", "mistranslation", "context_copied",
    "format_violation",
]

SYSTEM_PROMPT = """You are an expert evaluator of Korean-to-English dialogue translation.

A real-time translator translated ONE Korean utterance (the CURRENT TURN) of a casual spoken conversation into English. The translator may or may not have seen some of the earlier turns. You see the whole earlier conversation, who the speakers are, a gold reference translation, and specific checks written by the dataset author. Judge the CANDIDATE translation of the current turn only.

How to judge
- The reference is one good translation, not the only one. Accept any wording that conveys the same content, reference resolution, register and intent. Do not penalise paraphrase.
- Use the earlier conversation to decide what the Korean really means: who "걔/그 사람" is, what an omitted subject or object refers to, which sense a word has, whether a fragment continues an earlier turn, which term or name was used before.
- The candidate must translate only the current turn. Repeating or translating earlier turns, adding notes, explanations, alternatives, quotes around the output, labels such as "Translation:", or answering the speaker instead of translating are errors.
- The Korean current turn may be an incomplete fragment of a longer utterance. A faithful fragment-like translation is correct; inventing the missing continuation is an addition.
- Speaker personas (name, gender, age, role) are given to you only as ground truth for judging. The translator did not see them. Still judge the output against the truth: a wrong he/she is wrong.
- Evaluate each dimension independently. A severe problem in one dimension must not lower unrelated dimensions.
- Be strict and consistent. A score of 5 means there is nothing to fix.

Scores (integers 1-5)

meaning_preservation - is all content of the current Korean turn conveyed?
  5 all content and nuance conveyed
  4 a minor nuance or softener lost
  3 one meaningful element lost or distorted, main point intact
  2 main point partly wrong or several elements lost
  1 meaning mostly lost, wrong, or empty output

no_hallucination - does the candidate avoid adding content not supported by the current turn (or needed from context to make it explicit)?
  5 nothing unsupported added
  4 a small harmless embellishment
  3 an added phrase that changes emphasis or implies unstated facts
  2 an added clause or claim, or content copied from earlier turns
  1 mostly invented or mostly a translation of other turns

contextual_correctness - are context-dependent choices right given the earlier conversation (coreference, omitted arguments made explicit correctly, word sense, discourse connectives, continuation of fragments, consistency with earlier facts)?
  5 every context-dependent choice correct
  4 correct but one choice slightly vague where context allowed precision
  3 one context-dependent choice wrong or left ambiguous where it matters
  2 several wrong, or a key referent wrong
  1 contradicts the conversation
  If the turn has no context-dependent element, rate how well it fits the conversation.

speaker_consistency - are person references right: who does what, speaker vs. addressee vs. third party, gendered pronouns, names?
  5 all correct
  4 minor ambiguity that a reader resolves easily
  3 one person reference wrong or unclear
  2 a gender or who-did-what error that misleads
  1 persons systematically confused

register_consistency - does the tone match the speaker and relationship (banmal vs. jondaetmal, casual vs. formal, politeness, emotion), and the tone of earlier turns?
  5 fully appropriate
  4 slightly off (a bit stiff or a bit too casual)
  3 noticeably wrong level for this relationship
  2 clearly wrong (formal business tone among close friends, or rude to a superior)
  1 register opposite of the source

naturalness - is it fluent, idiomatic spoken English?
  5 sounds like a native speaker in this conversation
  4 fluent with a slightly unusual phrase
  3 understandable but awkward or translationese
  2 clumsy, hard to read
  1 broken or not English

error_labels - zero or more of these, only when clearly present:
  omission, addition, wrong_coreference, wrong_gender, wrong_register, lexical_inconsistency (a term, name or entity rendered differently from earlier turns), unnatural, mistranslation, context_copied (earlier turns repeated or translated), format_violation (anything other than a single plain translation: labels, notes, alternatives, multiple lines, quotes, answers)

checks - for EVERY check listed in the input, in the given order, decide pass true/false with a reason of at most 20 words. A check passes only if the candidate satisfies the requirement, allowing equivalent wording. An empty or off-target candidate fails every check.

Output: a single JSON object, no other text, exactly in this shape:
{"scores": {"meaning_preservation": 1-5, "no_hallucination": 1-5, "contextual_correctness": 1-5, "speaker_consistency": 1-5, "register_consistency": 1-5, "naturalness": 1-5},
 "error_labels": ["..."],
 "checks": [{"index": 0, "pass": true, "reason": "..."}],
 "rationale": "at most 50 words"}"""


def _speaker_line(label: str, speakers: dict) -> str:
    persona = speakers.get(label)
    return f"{label} ({persona})" if persona else label


def build_user_prompt(instance: dict, hypothesis: str) -> str:
    """Instance block first and candidate last, so pairs of one instance share a prefix."""
    speakers = instance.get("speakers") or {}
    lines = ["SPEAKERS:"]
    for label, persona in speakers.items():
        lines.append(f"- {label}: {persona}")
    lines.append("")
    lines.append("EARLIER CONVERSATION (oldest first; Korean source and gold English):")
    previous = instance.get("previous_turns") or []
    if not previous:
        lines.append("(none)")
    for turn in previous:
        lines.append(f"[turn {turn.get('turn_id')}] {turn.get('speaker', '?')}")
        lines.append(f"  KO: {turn.get('ko', '')}")
        lines.append(f"  EN: {turn.get('en', '')}")
    lines.append("")
    lines.append(f"CURRENT TURN [turn {instance.get('target_turn_id')}] "
                 f"{_speaker_line(instance.get('current_speaker', '?'), speakers)}")
    lines.append(f"  KO: {instance['current_source']}")
    lines.append(f"REFERENCE TRANSLATION: {instance['reference_translation']}")
    lines.append("")
    lines.append("CHALLENGE TAGS: " + ", ".join(instance.get("challenge_tags") or []))
    lines.append("CHECKS:")
    for index, check in enumerate(instance.get("checks") or []):
        lines.append(f"  {index}. [{check.get('tag')}] {check.get('requirement')}")
    lines.append("")
    lines.append("CANDIDATE TRANSLATION (between the markers, verbatim):")
    lines.append("<<<")
    lines.append(hypothesis)
    lines.append(">>>")
    return "\n".join(lines)


def validate(value, n_checks: int) -> dict:
    """Return a normalised verdict or raise ValueError with the reason."""
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    scores = value.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("missing scores object")
    clean_scores = {}
    for dim in DIMENSIONS:
        score = scores.get(dim)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score != int(score) \
                or not 1 <= score <= 5:
            raise ValueError(f"score {dim}={score!r} is not an integer 1-5")
        clean_scores[dim] = int(score)
    labels = value.get("error_labels", [])
    if not isinstance(labels, list) or any(label not in ERROR_LABELS for label in labels):
        raise ValueError(f"bad error_labels {labels!r}")
    checks = value.get("checks", [])
    if not isinstance(checks, list) or len(checks) != n_checks:
        raise ValueError(f"expected {n_checks} checks, got {checks!r}")
    by_index = {}
    for position, check in enumerate(checks):
        if not isinstance(check, dict) or not isinstance(check.get("pass"), bool):
            raise ValueError(f"bad check {check!r}")
        index = check.get("index", position)
        if not isinstance(index, int) or not 0 <= index < n_checks or index in by_index:
            raise ValueError(f"bad check index {index!r}")
        by_index[index] = {"index": index, "pass": check["pass"],
                           "reason": str(check.get("reason", ""))}
    rationale = value.get("rationale", "")
    return {"scores": clean_scores,
            "error_labels": sorted(set(labels), key=ERROR_LABELS.index),
            "checks": [by_index[i] for i in range(n_checks)],
            "rationale": str(rationale)}


class CostMeter:
    """Per-call usage log with a cumulative total that survives restarts."""

    def __init__(self, path: Path, model: str, budget: float, report_every: int = 10):
        if model not in PRICES:
            raise SystemExit(f"no price for {model!r}; add it to PRICES before spending money")
        self.path = path
        self.model = model
        self.price = PRICES[model]
        self.budget = budget
        self.report_every = report_every
        previous = common.read_jsonl(path)
        self.total = max((r.get("cumulative_cost", 0.0) for r in previous), default=0.0)
        self.previous_calls = len(previous)
        self.calls = 0
        self.session_cost = 0.0
        self.in_flight = 0

    def cost_of(self, usage: dict) -> tuple[float, dict]:
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        price_in, price_cached, price_out = self.price
        cost = ((prompt - cached) * price_in + cached * price_cached
                + completion * price_out) / 1e6
        return cost, {"prompt_tokens": prompt, "cached_tokens": cached,
                      "completion_tokens": completion, "reasoning_tokens": reasoning}

    def mean_call_cost(self) -> float:
        calls = self.previous_calls + self.calls
        return self.total / calls if calls else 0.02

    def may_start(self) -> bool:
        """Leave room for the calls already in flight plus this one."""
        projected = self.total + (self.in_flight + 1) * self.mean_call_cost()
        return projected <= self.budget

    def record(self, usage: dict, **fields) -> float:
        cost, tokens = self.cost_of(usage)
        self.total += cost
        self.session_cost += cost
        self.calls += 1
        common.append_jsonl(self.path, {
            "at": datetime.now(timezone.utc).isoformat(), "purpose": "context_judge",
            "model": self.model, "judge_version": JUDGE_VERSION, **fields, **tokens,
            "cost": round(cost, 6), "cumulative_cost": round(self.total, 6)})
        if self.calls % self.report_every == 0:
            print(f"[judge] {self.calls} calls this session, cumulative estimated cost "
                  f"${self.total:.4f} / budget ${self.budget:.2f}", flush=True)
        return cost


async def judge_pair(client, meter: CostMeter, args, pair: dict) -> dict | None:
    """Ask once, retry once on malformed JSON; API errors retry with backoff."""
    instance = pair["instance"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(instance, pair["hypothesis"])}]
    n_checks = len(instance.get("checks") or [])
    malformed = 0
    api_failures = 0
    last_error = None
    while malformed < 2 and api_failures < 4:
        started = time.time()
        try:
            response = await client.chat.completions.create(
                model=args.model, messages=messages,
                reasoning_effort=args.reasoning_effort,
                response_format={"type": "json_object"},
                max_completion_tokens=args.max_completion_tokens)
        except Exception as exc:
            api_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[judge] API error on {pair['key']}: {last_error}", flush=True)
            await asyncio.sleep(min(2 ** api_failures, 30) + random.random())
            continue
        usage = response.model_dump().get("usage") or {}
        content = response.choices[0].message.content or ""
        try:
            verdict = validate(json.loads(content), n_checks)
            ok, reason = True, None
        except (json.JSONDecodeError, ValueError) as exc:
            verdict, ok, reason = None, False, f"{type(exc).__name__}: {exc}"
        meter.record(usage, pair_key=pair["key"], instance_id=instance["instance_id"],
                     attempt=malformed + 1, ok=ok, error=reason,
                     latency_ms=round((time.time() - started) * 1000),
                     finish_reason=response.choices[0].finish_reason)
        if ok:
            return {"verdict": verdict, "raw": content}
        malformed += 1
        last_error = reason
        print(f"[judge] malformed output on {pair['key']}: {reason}", flush=True)
    print(f"[judge] giving up on {pair['key']}: {last_error}", flush=True)
    return None


async def run_pairs(pairs: list[dict], args, meter: CostMeter, cache_path: Path,
                    cache: dict) -> tuple[int, bool]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=common.env_value("OPENAI_API_KEY"))
    queue = list(pairs)
    done = 0
    stopped = False
    lock = asyncio.Lock()

    async def worker():
        nonlocal done, stopped
        while True:
            async with lock:
                if not queue or stopped:
                    return
                if not meter.may_start():
                    stopped = True
                    print(f"[judge] budget stop: cumulative ${meter.total:.4f}, "
                          f"{meter.in_flight} in flight, budget ${meter.budget:.2f}", flush=True)
                    return
                pair = queue.pop(0)
                meter.in_flight += 1
            try:
                result = await judge_pair(client, meter, args, pair)
            finally:
                meter.in_flight -= 1
            if result is None:
                continue
            entry = {"key": pair["key"], "judge_version": JUDGE_VERSION, "model": args.model,
                     "instance_id": pair["instance"]["instance_id"],
                     "hypothesis": pair["hypothesis"], **result["verdict"],
                     "raw_output": result["raw"]}
            cache[pair["key"]] = entry
            common.append_jsonl(cache_path, entry)
            done += 1

    await asyncio.gather(*(worker() for _ in range(max(1, args.concurrency))))
    return done, stopped


def fan_out(rows: list[dict], row_pair: dict, cache: dict, instances: dict) -> list[dict]:
    out = []
    for row in rows:
        entry = cache.get(row_pair[row["job_id"]])
        if entry is None:
            continue
        checks_meta = instances[row["instance_id"]].get("checks") or []
        checks = [{"index": c["index"], "tag": checks_meta[c["index"]].get("tag"),
                   "requirement": checks_meta[c["index"]].get("requirement"),
                   "pass": c["pass"], "reason": c["reason"]} for c in entry["checks"]]
        scores = entry["scores"]
        out.append({
            "job_id": row["job_id"],
            "judge_pair_key": entry["key"],
            "judge_model": entry["model"],
            "judge_version": entry["judge_version"],
            **{f"judge_{dim}": scores[dim] for dim in DIMENSIONS},
            "judge_mean": sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS),
            "judge_error_labels": entry["error_labels"],
            "judge_checks": checks,
            "judge_check_pass_rate": (sum(c["pass"] for c in checks) / len(checks)
                                      if checks else None),
            "judge_rationale": entry["rationale"],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instances")
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--budget-usd", type=float, default=12.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=4000)
    parser.add_argument("--limit", type=int, help="judge at most N new pairs (for testing)")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print the system + user prompt of the first pending pair")
    parser.add_argument("--dry-run", action="store_true",
                        help="count pairs and print the first prompt; no API calls")
    args = parser.parse_args()

    common.load_env()
    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, args.instances))
    if not instances:
        raise SystemExit("no instances found; pass --instances")
    rows = list(common.latest_by_job(common.read_jsonl(run_dir / "translations.jsonl")).values())
    rows = common.scorable_rows(rows, instances)

    cache_path = run_dir / "judge_cache.jsonl"
    cache = {c["key"]: c for c in common.read_jsonl(cache_path)}
    pairs: dict[str, dict] = {}
    row_pair = {}
    for row in rows:
        key = common.digest(JUDGE_VERSION, args.model, row["instance_id"], row["hypothesis"])
        row_pair[row["job_id"]] = key
        pairs.setdefault(key, {"key": key, "instance": instances[row["instance_id"]],
                               "hypothesis": row["hypothesis"]})
    pending = [p for k, p in pairs.items() if k not in cache]
    # Same-instance pairs back to back: they share the longest prompt prefix.
    pending.sort(key=lambda p: (p["instance"]["instance_id"], p["hypothesis"]))
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[judge] {len(rows)} rows, {len(pairs)} unique (instance, hypothesis) pairs, "
          f"{len(pairs) - sum(k not in cache for k in pairs)} cached, "
          f"{len(pending)} to judge now", flush=True)

    if (args.show_prompt or args.dry_run) and pending:
        first = pending[0]
        print("=" * 30 + " SYSTEM " + "=" * 30)
        print(SYSTEM_PROMPT)
        print("=" * 30 + " USER " + "=" * 32)
        print(build_user_prompt(first["instance"], first["hypothesis"]))
        print("=" * 68, flush=True)
    if args.dry_run:
        return

    meter = CostMeter(run_dir / "judge_usage.jsonl", args.model, args.budget_usd)
    print(f"[judge] cumulative cost so far ${meter.total:.4f} over {meter.previous_calls} "
          f"calls; budget ${args.budget_usd:.2f}", flush=True)
    stopped = False
    if pending:
        if not common.env_value("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY missing (.env)")
        done, stopped = asyncio.run(run_pairs(pending, args, meter, cache_path, cache))
        print(f"[judge] judged {done} pairs this session; session cost "
              f"${meter.session_cost:.4f}; cumulative ${meter.total:.4f}", flush=True)

    out = fan_out(rows, row_pair, cache, instances)
    common.write_jsonl(run_dir / "scores_judge.jsonl", out)
    missing = len(rows) - len(out)
    print(f"[judge] wrote {len(out)} rows to scores_judge.jsonl"
          + (f"; {missing} rows still unjudged" if missing else ""), flush=True)
    if stopped:
        sys.exit(3)


if __name__ == "__main__":
    main()
