"""LLM labeler for emission schedules (prefix14).

For each source sentence, ask a model for a schedule [{"after": k, "text": "..."}]: "once source word k has
been heard, append text". Each reply goes through the mechanical gates in schedule_gate.check_schedule; on a
failure the model is asked once more with the reason; a second failure is a FAIL (not written). Hy word-prefix
translations (hy_word_prefixes.py) supply the "known content after word k" sets used by the gate.

Output: --out jsonl {"sid","lang","tgt","schedule":[[after,text],...],"attempts":n}; per-call usage (tokens,
est. cost) appends to <out>.usage.jsonl; --budget counts what that file already holds and aborts past the cap
(exit code 2). API errors (429/5xx/connection) are retried 3x; exceptions on more than 1% of rows exit 3.
Resume: sids already in --out or in <out>.fail.jsonl (gate FAILs, with the reason) are skipped, so a rerun
retries only ERROR rows and pays nothing twice. Endpoint: Letsur gateway (LETSUR_API_KEY).

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
from schedule_gate import (check_schedule, clause_ends, fix_leading_punct, merge_same_after, parse_covers,  # noqa: E402
                           parse_schedule, shape_check, words)

PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-5-mini": (0.25, 2.00)}
LANG_NAME = {"en": "English", "ko": "Korean"}
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "schedule_rules.txt")


DEFERRAL_CAP = {"en": 8, "ko": 6}
RULES = {"path": RULES_PATH}


def n_cap(lang):
    """Deferral cap in source words; 0 (--deferral-cap 0) turns the rule off (the clause rule still holds)."""
    return DEFERRAL_CAP[lang] or 10**6


def should_stop(spent, budget):
    return spent > budget


def spent_so_far(usage_path):
    """Sum of est_usd already logged in <out>.usage.jsonl, so a rerun keeps counting against --budget."""
    if not os.path.exists(usage_path):
        return 0.0
    total = 0.0
    for l in open(usage_path, encoding="utf-8"):
        try:
            total += float(json.loads(l).get("est_usd") or 0.0)
        except (ValueError, AttributeError):
            continue  # a line cut by a crash
    return total


def retryable(e):
    """429, 5xx and connection/timeout errors are worth retrying; other API errors are not."""
    import openai
    if isinstance(e, (openai.APIConnectionError, openai.RateLimitError)):  # APITimeoutError is a connection error
        return True
    return isinstance(e, openai.APIStatusError) and (e.status_code == 429 or e.status_code >= 500)


def with_retry(fn, is_retryable=retryable, delays=(2, 4, 8), sleep=time.sleep):
    """fn(), retried after each delay in `delays` while the error is retryable; the last error is raised."""
    for d in (*delays, None):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if d is None or not is_retryable(e):
                raise
            sleep(d)


def too_many_errors(errors, n_todo):
    """ERROR (exceptions, not gate FAILs) above 1% of the attempted rows fails the run."""
    return errors > 0.01 * n_todo


def build_prompt(row, note=""):
    tokens = words(row["src"], row["lang"])
    numbered = " ".join(f"{i}:{t}" for i, t in enumerate(tokens, 1))
    tgt_name = LANG_NAME[row["tgt"]]
    rules = open(RULES["path"], encoding="utf-8").read().strip().format(N=n_cap(row["lang"]))
    lines = [f"Source sentence ({LANG_NAME[row['lang']]}), words numbered from 1:", numbered]
    if row.get("ref"):
        lines.append(f"Reference translation ({tgt_name}): {row['ref']}")
    lines.append(f"Write a simultaneous-interpretation emission schedule into {tgt_name}.")
    lines.append(rules)
    if note:
        lines.append(note)
    lines.append('Answer with JSON only: {"schedule": [{"after": k, "text": "...", "src": [i, j, ...]}, ...]} where '
                 '"src" lists the numbers of the source words that "text" translates (all <= "after"; [] for an empty text).')
    return "\n".join(lines)


def gate(row, hy_row, reply):
    """Reason string if `reply` (parsed JSON) is rejected, else None."""
    tokens = words(row["src"], row["lang"])
    known = known_sets(hy_row["prefixes"], row["tgt"])
    try:
        items = parse_schedule(reply)
        covers = parse_covers(reply)
    except ValueError as e:
        return str(e)
    why = check_schedule(items, tokens, row["lang"], row["tgt"], known_at=lambda k: known[k - 1],
                         full_known=known[-1], N=n_cap(row["lang"]), clause_end=clause_ends(tokens, row["lang"]),
                         covers=covers)
    return why or shape_check(items, row["tgt"])


def hy_disagreements(row, hy_row, reply):
    """How many non-empty items the Hy-stem lookahead rule would have rejected: a diagnostic on the
    self-reported alignment (recorded, never a FAIL)."""
    from build_prefix import content
    known = known_sets(hy_row["prefixes"], row["tgt"])
    n = 0
    for after, text in parse_schedule(reply):
        if text.strip() and (content(text, row["tgt"]) - known[after - 1]) & known[-1]:
            n += 1
    return n


def reviewer_notes(judge_path):
    """sid -> first-request note for the rows a judge file (judge_schedule.py) failed."""
    notes = {}
    for l in open(judge_path, encoding="utf-8"):
        if not l.strip():
            continue
        j = json.loads(l)
        if not j.get("pass"):
            notes[j["sid"]] = f"A reviewer found: {j.get('note') or ', '.join(j.get('defects') or [])}. Fix that."
    return notes


def label_one(row, hy_row, ask, fail=None, note=""):
    """Output record, or None on FAIL. ask(row, note) -> parsed JSON dict. `fail(reason)` is told why.
    `note` (a reviewer's finding) is appended to the first request; a gate rejection replaces it on the retry."""
    fail = fail or (lambda why: None)
    tokens = words(row["src"], row["lang"])
    prefixes = (hy_row or {}).get("prefixes")
    if not prefixes or len(prefixes) != len(tokens):
        fail("no hy prefixes")
        return None
    why = None
    for attempt in (1, 2):
        reply = ask(row, note)
        why = gate(row, hy_row, reply)
        if why is None:
            items, covers = merge_same_after(fix_leading_punct(parse_schedule(reply)), parse_covers(reply))
            rec = {"sid": row["sid"], "lang": row["lang"], "tgt": row["tgt"],
                   "schedule": [[k, t] for k, t in items], "attempts": attempt}
            if any(c is not None for c in covers):
                rec["src_words"] = [sorted(c) if c is not None else None for c in covers]
                rec["hy_disagree"] = hy_disagreements(row, hy_row, reply)
            return rec
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
    ap.add_argument("--rules", default=RULES_PATH, help="rules text appended to every prompt")
    ap.add_argument("--only-sids", help="file with one sid per line: label only these")
    ap.add_argument("--notes", help="judge_schedule.py output: its failed rows' notes go into the first request")
    ap.add_argument("--deferral-cap", type=int, default=None,
                    help="deferral cap in source words for both languages (0 = off); default en 8 / ko 6")
    a = ap.parse_args()
    RULES["path"] = a.rules
    if a.deferral_cap is not None:
        DEFERRAL_CAP["en"] = DEFERRAL_CAP["ko"] = a.deferral_cap

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
    fail_path = a.out + ".fail.jsonl"
    if os.path.exists(fail_path):
        done |= {json.loads(l)["sid"] for l in open(fail_path, encoding="utf-8") if l.strip()}
    out = open(a.out, "a", encoding="utf-8")
    fail_out = open(fail_path, "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    lock = threading.Lock()
    st = {"spent": spent_so_far(a.out + ".usage.jsonl"), "ok": 0, "fail": 0, "error": 0, "seen": 0}
    print(f"already spent ${st['spent']:.2f} (from {a.out}.usage.jsonl)", flush=True)

    def ask(row, note=""):
        t0 = time.time()
        resp = with_retry(lambda: client.chat.completions.create(
            model=a.model, reasoning_effort=a.reasoning,
            messages=[{"role": "user", "content": build_prompt(row, note)}]))
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
            rec = label_one(row, hy.get(row["sid"]), ask, fail=reasons.append, note=notes.get(row["sid"], ""))
        except Exception as e:  # noqa: BLE001
            with lock:
                st["seen"] += 1
                st["error"] += 1
                print(f"ERROR {row['sid']}: {e}", flush=True)
            return
        with lock:
            st["seen"] += 1
            if rec is None:
                st["fail"] += 1
                print(f"FAIL {row['sid']}: {reasons[-1] if reasons else '?'}", flush=True)
                fail_out.write(json.dumps({"sid": row["sid"], "reason": reasons[-1] if reasons else "?"}, ensure_ascii=False) + "\n")
                fail_out.flush()
            else:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                st["ok"] += 1
            if st["seen"] % 50 == 0:
                print(f"{st['seen']}/{len(todo)} ok={st['ok']} fail={st['fail']} error={st['error']} "
                      f"est ${st['spent']:.2f}", flush=True)

    notes = reviewer_notes(a.notes) if a.notes else {}
    if a.only_sids:
        only = {l.strip() for l in open(a.only_sids, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["sid"] in only]
    todo = [r for r in rows if r["sid"] not in done]
    print(f"rows {len(rows)}, todo {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    if should_stop(st["spent"], a.budget):
        print(f"BUDGET_STOP ${st['spent']:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"LABEL_SCHEDULE_DONE ok={st['ok']} fail={st['fail']} error={st['error']} est ${st['spent']:.2f}", flush=True)
    if too_many_errors(st["error"], len(todo)):
        print(f"LABEL_SCHEDULE_ERRORS {st['error']} > 1% of {len(todo)} - rerun to retry them", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
