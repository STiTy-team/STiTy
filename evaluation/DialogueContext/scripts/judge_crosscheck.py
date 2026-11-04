"""Cross-family re-judge of a sample of DialogueContext verdicts with a Gemini model.

The main judge (judge_context.py, gpt-6-sol) and one candidate (gpt-6-luna) are both OpenAI
models. This script re-judges a stratified sample of the same (instance, hypothesis) pairs
with a Gemini model, using the identical system and user prompt from judge_context.py, so
the two judges can be compared for self-preference bias.

Sample (seed 20260925): for each of the 20 instances, four rounds pick one unjudged pair
produced by a rotating model, so every model gets 20 pairs. Rounds 0-1 are the 40-pair
human-audit sample ("audit"), rounds 2-3 the 40 extra pairs ("extra").

Cost tracking (.claude/rules/cost-watch.md):
  - every API call appends usage, per-call cost and cumulative cost to crosscheck_usage.jsonl
  - the cumulative cost resumes from that file; no call starts once the budget would be crossed
  - finished verdicts are appended to crosscheck_cache.jsonl (resume-safe)

    venv-tf5/bin/python evaluation/DialogueContext/scripts/judge_crosscheck.py \
        --run-dir evaluation/DialogueContext/results/dctx-20260925 [--budget-usd 3]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring_common as common  # noqa: E402
from judge_context import JUDGE_VERSION, SYSTEM_PROMPT, build_user_prompt, validate  # noqa: E402

CROSSCHECK_VERSION = f"{JUDGE_VERSION}+gemini-xc.1"
SEED = 20260925
MODELS = ["deepl-quality", "gemma3-4b", "gpt-6-luna", "qwen3.5-4b"]
# USD per 1M tokens (input, cached input, output incl. thinking); Gemini API pricing page,
# paid tier, read 2026-09-26 (3.8 Flash promotional price through 2026-12-31).
PRICES = {
    "gemini-3.8-flash": (0.75, 0.075, 3.75),
    "gemini-2.5-pro": (1.25, 1.25, 10.00),
    "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
}
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def draw_sample(run_dir: Path) -> list[dict]:
    merged = common.read_jsonl(run_dir / "merged.jsonl")
    cache = {(c["instance_id"], c["hypothesis"]): c
             for c in common.read_jsonl(run_dir / "judge_cache.jsonl")}
    producers = defaultdict(set)
    for row in merged:
        key = (row["instance_id"], row.get("hypothesis"))
        if key in cache and row.get("judge_checks"):
            producers[key].add(row["model"])
    instances = sorted({k[0] for k in producers})
    rng = random.Random(SEED)
    chosen, used = [], set()
    for rnd in range(4):
        for i, inst in enumerate(instances):
            want = MODELS[(2 * i + rnd + rnd // 2) % 4]
            cands = sorted((k for k in producers if k[0] == inst and want in producers[k]
                            and cache[k]["key"] not in used), key=lambda k: k[1])
            if not cands:
                cands = sorted((k for k in producers if k[0] == inst
                                and cache[k]["key"] not in used), key=lambda k: k[1])
                want = "(fallback)" + want
            k = rng.choice(cands)
            used.add(cache[k]["key"])
            chosen.append({"key": cache[k]["key"], "instance_id": inst, "hypothesis": k[1],
                           "sample_model": want, "models": sorted(producers[k]),
                           "set": "audit" if rnd < 2 else "extra"})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--thinking-level", default="low")
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.model not in PRICES:
        raise SystemExit(f"no price for {args.model!r}; add it before spending money")

    import httpx

    run_dir = args.run_dir
    instances = common.load_instances(common.resolve_instances_path(run_dir, None))
    sample = draw_sample(run_dir)
    common.write_jsonl(run_dir / "crosscheck_sample.jsonl", sample)
    usage_path = run_dir / "crosscheck_usage.jsonl"
    cache_path = run_dir / "crosscheck_cache.jsonl"
    usage_rows = common.read_jsonl(usage_path)
    total = max((r.get("cumulative_cost", 0.0) for r in usage_rows), default=0.0)
    calls = len(usage_rows)
    cache = {c["key"]: c for c in common.read_jsonl(cache_path)
             if c.get("crosscheck_version") == CROSSCHECK_VERSION and c.get("model") == args.model}
    pending = [s for s in sample if s["key"] not in cache]
    n_cached = len(sample) - len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"[xc] sample {len(sample)}, cached {n_cached}, to judge {len(pending)}; "
          f"cumulative ${total:.4f}, budget ${args.budget_usd:.2f}", flush=True)

    key_value = common.env_value("GEMINI_API_KEY")
    if pending and not key_value:
        raise SystemExit("GEMINI_API_KEY missing (.env)")
    price_in, price_cached, price_out = PRICES[args.model]
    client = httpx.Client(timeout=180)
    for n, item in enumerate(pending, 1):
        mean = total / calls if calls else 0.02
        if total + mean > args.budget_usd:
            print(f"[xc] budget stop at ${total:.4f}", flush=True)
            sys.exit(3)
        instance = instances[item["instance_id"]]
        n_checks = len(instance.get("checks") or [])
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user",
                          "parts": [{"text": build_user_prompt(instance, item["hypothesis"])}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "maxOutputTokens": 8000,
                                 "thinkingConfig": {"thinkingLevel": args.thinking_level}},
        }
        verdict = None
        for attempt in range(1, 5):
            started = time.time()
            try:
                resp = client.post(API.format(model=args.model), json=body,
                                   headers={"x-goog-api-key": key_value})
            except httpx.HTTPError as exc:
                print(f"[xc] network error {item['key']}: {exc}", flush=True)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                print(f"[xc] HTTP {resp.status_code} {item['key']}: {resp.text[:300]}", flush=True)
                time.sleep(2 ** attempt + random.random())
                continue
            data = resp.json()
            meta = data.get("usageMetadata") or {}
            prompt = meta.get("promptTokenCount", 0) or 0
            cached = meta.get("cachedContentTokenCount", 0) or 0
            out = (meta.get("candidatesTokenCount", 0) or 0) + (meta.get("thoughtsTokenCount", 0) or 0)
            cost = ((prompt - cached) * price_in + cached * price_cached + out * price_out) / 1e6
            total += cost
            calls += 1
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])
                           if not p.get("thought"))
            try:
                verdict = validate(json.loads(text), n_checks)
                ok, err = True, None
            except (json.JSONDecodeError, ValueError) as exc:
                ok, err = False, f"{type(exc).__name__}: {exc}"
            common.append_jsonl(usage_path, {
                "at": datetime.now(timezone.utc).isoformat(), "purpose": "crosscheck_judge",
                "model": args.model, "crosscheck_version": CROSSCHECK_VERSION,
                "pair_key": item["key"], "instance_id": item["instance_id"], "attempt": attempt,
                "ok": ok, "error": err, "latency_ms": round((time.time() - started) * 1000),
                "finish_reason": cand.get("finishReason"), "prompt_tokens": prompt,
                "cached_tokens": cached, "candidates_tokens": meta.get("candidatesTokenCount", 0),
                "thoughts_tokens": meta.get("thoughtsTokenCount", 0),
                "price_per_1m": {"input": price_in, "cached": price_cached, "output": price_out},
                "cost": round(cost, 6), "cumulative_cost": round(total, 6)})
            if ok:
                break
            print(f"[xc] malformed {item['key']}: {err}", flush=True)
            verdict = None
        if verdict is None:
            print(f"[xc] giving up on {item['key']}", flush=True)
            continue
        entry = {"key": item["key"], "crosscheck_version": CROSSCHECK_VERSION, "model": args.model,
                 "instance_id": item["instance_id"], "hypothesis": item["hypothesis"], **verdict,
                 "raw_output": text}
        cache[item["key"]] = entry
        common.append_jsonl(cache_path, entry)
        if n % 10 == 0:
            print(f"[xc] {n}/{len(pending)} done, cumulative ${total:.4f}", flush=True)

    # Joined output: both judges side by side for every sampled pair.
    gpt = {c["key"]: c for c in common.read_jsonl(run_dir / "judge_cache.jsonl")}
    dims = list(gpt[sample[0]["key"]]["scores"])
    out = []
    for item in sample:
        g, x = gpt[item["key"]], cache.get(item["key"])
        if x is None:
            continue
        out.append({**item, "gpt_scores": g["scores"], "gemini_scores": x["scores"],
                    "gpt_mean": sum(g["scores"][d] for d in dims) / len(dims),
                    "gemini_mean": sum(x["scores"][d] for d in dims) / len(dims),
                    "gpt_checks": [c["pass"] for c in g["checks"]],
                    "gemini_checks": [c["pass"] for c in x["checks"]],
                    "gemini_check_reasons": [c["reason"] for c in x["checks"]],
                    "check_tags": [c.get("tag") for c in instances[item["instance_id"]]["checks"]],
                    "gemini_error_labels": x["error_labels"], "gemini_model": x["model"]})
    common.write_jsonl(run_dir / "crosscheck_judge.jsonl", out)
    print(f"[xc] wrote {len(out)} rows to crosscheck_judge.jsonl; cumulative ${total:.4f}", flush=True)


if __name__ == "__main__":
    main()
