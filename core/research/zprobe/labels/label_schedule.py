"""LLM labeler for emission schedules (prefix14).

For each source sentence, ask a model for a schedule [{"after": k, "text": "..."}]: "once source word k has
been heard, append text". Each reply goes through the mechanical gates in schedule_gate.check_schedule; on a
failure the model is asked once more with the reason; a second failure is a FAIL (not written). Hy word-prefix
translations (hy_word_prefixes.py) supply the "known content after word k" sets used by the gate.

Output: --out jsonl {"sid","lang","tgt","schedule":[[after,text],...],"attempts":n}; per-call usage (tokens,
est. cost) appends to <out>.usage.jsonl; --budget aborts past the cap (exit code 2). Resume: sids already in
--out are skipped. Endpoint: Letsur gateway (LETSUR_API_KEY).

usage: .venv/bin/python core/research/zprobe/labels/label_schedule.py --limit 50 \
    --out models/zprobe/data/prefix14/schedule_pilot.jsonl
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hy_word_prefixes import known_sets  # noqa: E402
from schedule_gate import check_schedule, clause_ends, parse_schedule, words  # noqa: E402

PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-5-mini": (0.25, 2.00)}
LANG_NAME = {"en": "English", "ko": "Korean"}
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "schedule_rules.txt")


def n_cap(lang):
    return 8 if lang == "en" else 6


def should_stop(spent, budget):
    return spent > budget


def build_prompt(row, note=""):
    tokens = words(row["src"], row["lang"])
    numbered = " ".join(f"{i}:{t}" for i, t in enumerate(tokens, 1))
    tgt_name = LANG_NAME[row["tgt"]]
    rules = open(RULES_PATH, encoding="utf-8").read().strip().format(N=n_cap(row["lang"]))
    lines = [f"Source sentence ({LANG_NAME[row['lang']]}), words numbered from 1:", numbered]
    if row.get("ref"):
        lines.append(f"Reference translation ({tgt_name}): {row['ref']}")
    lines.append(f"Write a simultaneous-interpretation emission schedule into {tgt_name}.")
    lines.append(rules)
    if note:
        lines.append(note)
    lines.append('Answer with JSON only: {"schedule": [{"after": k, "text": "..."}, ...]}')
    return "\n".join(lines)


def gate(row, hy_row, reply):
    """Reason string if `reply` (parsed JSON) is rejected, else None."""
    tokens = words(row["src"], row["lang"])
    known = known_sets(hy_row["prefixes"], row["tgt"])
    try:
        items = parse_schedule(reply)
    except ValueError as e:
        return str(e)
    return check_schedule(items, tokens, row["lang"], row["tgt"], known_at=lambda k: known[k - 1],
                          full_known=known[-1], N=n_cap(row["lang"]), clause_end=clause_ends(tokens, row["lang"]))


def label_one(row, hy_row, ask, fail=None):
    """Output record, or None on FAIL. ask(row, note) -> parsed JSON dict. `fail(reason)` is told why."""
    fail = fail or (lambda why: None)
    tokens = words(row["src"], row["lang"])
    prefixes = (hy_row or {}).get("prefixes")
    if not prefixes or len(prefixes) != len(tokens):
        fail("no hy prefixes")
        return None
    note, why = "", None
    for attempt in (1, 2):
        reply = ask(row, note)
        why = gate(row, hy_row, reply)
        if why is None:
            items = parse_schedule(reply)
            return {"sid": row["sid"], "lang": row["lang"], "tgt": row["tgt"],
                    "schedule": [[k, t] for k, t in items], "attempts": attempt}
        note = f"Previous attempt was rejected: {why}. Fix that."
    fail(why)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--hy", default="models/zprobe/data/prefix14/hy_prefixes.jsonl")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])

    hy = {}
    for l in open(a.hy, encoding="utf-8"):
        r = json.loads(l)
        hy[r["sid"]] = r
    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
    out = open(a.out, "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    lock = threading.Lock()
    st = {"spent": 0.0, "ok": 0, "fail": 0, "seen": 0}

    def ask(row, note=""):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=a.model, reasoning_effort=a.reasoning,
            messages=[{"role": "user", "content": build_prompt(row, note)}])
        u = resp.usage
        ec = (resp.model_extra or {}).get("estimated_cost") or {}
        try:
            cost = float(ec.get("amount"))
        except (TypeError, ValueError):
            pin, pout = PRICE.get(a.model, (0.25, 2.00))
            cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
        with lock:
            st["spent"] += cost
            usage.write(json.dumps({"sid": row["sid"], "model": a.model, "in": u.prompt_tokens,
                                    "out": u.completion_tokens, "est_usd": round(cost, 6),
                                    "sec": round(time.time() - t0, 1)}) + "\n")
            usage.flush()
        m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}  # gate reports "format: ..." and the retry note asks again

    def work(row):
        if should_stop(st["spent"], a.budget):
            return
        reasons = []
        try:
            rec = label_one(row, hy.get(row["sid"]), ask, fail=reasons.append)
        except Exception as e:  # noqa: BLE001
            with lock:
                st["seen"] += 1
                st["fail"] += 1
                print(f"ERROR {row['sid']}: {e}", flush=True)
            return
        with lock:
            st["seen"] += 1
            if rec is None:
                st["fail"] += 1
                print(f"FAIL {row['sid']}: {reasons[-1] if reasons else '?'}", flush=True)
            else:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                st["ok"] += 1
            if st["seen"] % 50 == 0:
                print(f"{st['seen']}/{len(rows)} ok={st['ok']} fail={st['fail']} est ${st['spent']:.2f}", flush=True)

    todo = [r for r in rows if r["sid"] not in done]
    print(f"rows {len(rows)}, todo {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    if should_stop(st["spent"], a.budget):
        print(f"BUDGET_STOP ${st['spent']:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"LABEL_SCHEDULE_DONE ok={st['ok']} fail={st['fail']} est ${st['spent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
