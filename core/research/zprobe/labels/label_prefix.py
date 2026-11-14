"""Prefix labeler (prefix16): the labeler sees only what the model will see.

For every source sentence, a mechanical list of candidate points (prefix_points.candidate_points) is
walked in order. At each point the LLM gets the source words heard so far (1..k) and the translation
committed so far, and answers {"add": text or "", "src": [source word numbers]}. Each answer passes a
step gate (range, lookahead against k, open-piece shape); a lookahead violation becomes a wait (empty
item) at once, a shape/format violation is re-asked once at that point and then becomes a wait. When the
sentence is over, a regret check shows the LLM the whole source (and the reference, if any) with the
piece boundaries and asks which pieces should not have gone out; a regretted piece becomes a wait and
the points after it are re-asked. The record keeps one item per candidate point, empty ones included -
those are the wait labels.

Output: --out jsonl {"sid","lang","tgt","schedule":[[after,text],...],"src_words":[[...],...],"attempts",
"regret_rounds","calls"}; <out>.fail.jsonl {"sid","reason"}; <out>.usage.jsonl one line per call with
kind "step" or "regret" (cost-watch). Resume: sids in --out or .fail.jsonl are skipped; a sentence cut by
a crash restarts from its first point. Exit 2 = budget stop, 3 = API errors on > 1% of rows.

usage: PYTHONPATH=. .venv/bin/python core/research/zprobe/labels/label_prefix.py --limit 50 \
    --sentences models/zprobe/data/prefix14/sentences.jsonl --out models/zprobe/data/prefix16/pilot.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from label_schedule import LANG_NAME  # noqa: E402
from schedule_gate import words  # noqa: E402

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "prefix_rules_v1.txt")
RULES = {"path": RULES_PATH}

END_NOTE = "The source sentence ENDS here: emit everything that is still missing and close the sentence."
MORE_NOTE = "More source may follow; you do not know how much."


def build_step_prompt(row: dict, k: int, committed: str, is_end: bool, note: str = "") -> str:
    """Prompt for one decision point: words 1..k, the committed text, the rules. Never the reference, never
    words after k."""
    tokens = words(row["src"], row["lang"])[:k]
    numbered = " ".join(f"{i}:{t}" for i, t in enumerate(tokens, 1))
    rules = open(RULES["path"], encoding="utf-8").read().strip()
    lines = [f"Source ({LANG_NAME[row['lang']]}) heard so far, words numbered from 1:", numbered,
             END_NOTE if is_end else MORE_NOTE,
             f"Translation ({LANG_NAME[row['tgt']]}) committed so far: {committed.strip() or '(nothing yet)'}",
             rules]
    if note:
        lines.append(note)
    lines.append('Answer with JSON only: {"add": "...", "src": [i, j, ...]} - "src" lists the numbers of the heard '
                 'source words that "add" translates (all <= the last word shown); "add": "" with "src": [] when '
                 'nothing can be added yet.')
    return "\n".join(lines)


def build_regret_prompt(row: dict, pieces: list[tuple[int, str]]) -> str:
    """After the sentence: whole source, reference if any, the pieces numbered - which ones were a mistake?"""
    numbered = " ".join(f"[{i}] {t.strip()}" for i, (_, t) in enumerate(pieces, 1))
    lines = [f"A simultaneous interpreter produced this translation piece by piece, each piece committed before "
             f"hearing the rest of the source. The pieces are numbered in order.",
             f"Source ({LANG_NAME[row['lang']]}): {row['src']}"]
    if row.get("ref"):
        lines.append(f"Reference translation ({LANG_NAME[row['tgt']]}), for meaning only: {row['ref']}")
    lines += [f"Pieces ({LANG_NAME[row['tgt']]}): {numbered}",
              "Knowing the whole source now, list the pieces that should NOT have been emitted at that moment: "
              "(i) the piece left no natural place for content that came later (word order is now blocked), "
              "(ii) its meaning turned out wrong once the rest was heard (negation, tense, what a modifier attaches to), "
              "(iii) it contains content that had not been heard yet. A piece that is merely awkward but correct is not "
              "a regret. The last piece may be listed only for (ii) or (iii).",
              'Answer with JSON only: {"regret": [piece numbers], "note": "one short sentence, or empty"}']
    return "\n".join(lines)


def parse_add(reply) -> tuple[str, set[int]]:
    """{"add": str, "src": [int]} -> (text, src set); ValueError("format: ...") otherwise. Blank add -> ("", set())."""
    if not isinstance(reply, dict) or "add" not in reply:
        raise ValueError("format: expected {'add': ..., 'src': [...]}")
    add = reply["add"]
    if not isinstance(add, str):
        raise ValueError(f"format: 'add' must be a string: {add!r}")
    if not add.strip():
        return "", set()
    src = reply.get("src")
    if not isinstance(src, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in src):
        raise ValueError(f"format: 'src' must be a list of ints: {src!r}")
    return add.strip(), set(src)


def parse_regret(reply, n_pieces: int) -> tuple[list[int], str]:
    """{"regret": [int], "note": str} -> (piece numbers within 1..n, note); ValueError on bad shape."""
    if not isinstance(reply, dict) or not isinstance(reply.get("regret"), list):
        raise ValueError("format: expected {'regret': [...], 'note': ...}")
    nums = sorted({x for x in reply["regret"] if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n_pieces})
    return nums, str(reply.get("note") or "")


# --- the per-sentence loop ---------------------------------------------------------------------------------------
from prefix_points import candidate_points  # noqa: E402
from schedule_gate import fix_leading_punct, join_texts, shape_check_open  # noqa: E402

END_REASK = "The sentence has ended; emit all that is still missing."


def step_gate(text: str, cov: set, k: int, items, covers, tgt: str, tokens: list[str], is_end: bool) -> str | None:
    """Reason if this point's answer is rejected: range (text without src, src < 1), lookahead (src > k),
    then the open-piece shape of the pieces so far plus this one."""
    if text and not cov:
        return "range: text without src"
    if any(x < 1 for x in cov):
        return f"range: src words {sorted(x for x in cov if x < 1)} below 1"
    late = sorted(x for x in cov if x > k)
    if late:
        return f"lookahead: src words {late} are after word {k}"
    if not text:
        return None
    return shape_check_open(list(items) + [(k, text)], list(covers) + [set(cov)], tgt, tokens, final=is_end)


def label_sentence(row: dict, ask_step, ask_regret, *, max_regret: int = 2, fail=None, note: str = "",
                   min_gap: int = 2) -> dict | None:
    """Walk the candidate points, gate each answer, then run the regret check (see the module docstring).
    ask_step(row, k, committed, is_end, note) and ask_regret(row, pieces) return parsed JSON dicts.
    Returns the label record, or None after fail(reason)."""
    fail = fail or (lambda why: None)
    tokens = words(row["src"], row["lang"])
    points = candidate_points(tokens, row["lang"], min_gap)
    items: list[tuple[int, str]] = []
    covers: list[set] = []
    st = {"calls": 0, "attempts": 0}

    def ask_point(k, is_end, n):
        """(text, cov) for this point, or None when the sentence fails here."""
        committed = join_texts([t for _, t in items])
        for attempt in (1, 2):
            st["calls"] += 1
            reply = ask_step(row, k, committed, is_end, n)
            try:
                text, cov = parse_add(reply)
                why = step_gate(text, cov, k, items, covers, row["tgt"], tokens, is_end)
            except ValueError as e:
                text, cov, why = "", set(), str(e)
            if why is None and is_end and not text:
                why = "final point empty"
            if why is None:
                return text, cov
            if why.startswith("lookahead"):
                return "", set()  # not re-asked: unheard content is a wait by definition
            if attempt == 1:
                st["attempts"] += 1
                hint = END_REASK if why == "final point empty" else f"Previous attempt was rejected: {why}. Fix that."
                n = f"{n} {hint}" if n else hint
                continue
            if is_end:
                fail(why if why == "final point empty" else f"final point: {why}")
                return None
            return "", set()

    def run_from(start, first_note):
        del items[start:]
        del covers[start:]
        for idx in range(start, len(points)):
            res = ask_point(points[idx], idx == len(points) - 1, first_note if idx == start else "")
            if res is None:
                return False
            items.append((points[idx], res[0]))
            covers.append(set(res[1]))
        return True

    if not run_from(0, note):
        return None
    rounds = 0
    while True:
        pieces = [(k, t) for k, t in items if t.strip()]
        try:
            regs, rnote = parse_regret(ask_regret(row, pieces), len(pieces))
        except ValueError:
            regs, rnote = [], ""  # a malformed verdict is no verdict
        if not regs:
            break
        if rounds >= max_regret:
            fail("regret unresolved")
            return None
        rounds += 1
        idx = items.index(pieces[regs[0] - 1])
        if idx == len(items) - 1:
            ok = run_from(idx, f"A reviewer found: {rnote}. Fix that." if rnote else "A reviewer rejected the last piece. Fix that.")
        else:
            items[idx] = (items[idx][0], "")
            covers[idx] = set()
            ok = run_from(idx + 1, "")
        if not ok:
            return None
    fixed = fix_leading_punct(items)
    return {"sid": row["sid"], "lang": row["lang"], "tgt": row["tgt"],
            "schedule": [[k, t] for k, t in fixed], "src_words": [sorted(c) for c in covers],
            "attempts": st["attempts"], "regret_rounds": rounds, "calls": st["calls"]}


# --- CLI ------------------------------------------------------------------------------------------------------
import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from label_schedule import PRICE, reviewer_notes, should_stop, spent_so_far, too_many_errors, with_retry  # noqa: E402


def plan_todo(rows, out_path):
    """(rows not yet in --out or its .fail.jsonl, est_usd already spent per the usage file)."""
    done = set()
    for p in (out_path, out_path + ".fail.jsonl"):
        if os.path.exists(p):
            done |= {json.loads(l)["sid"] for l in open(p, encoding="utf-8") if l.strip()}
    return [r for r in rows if r["sid"] not in done], spent_so_far(out_path + ".usage.jsonl")


def make_askers(client, model, reasoning, usage, lock, st):
    """(ask_step, ask_regret) for label_sentence: each call goes through with_retry, logs usage (kind step|regret)
    and returns the parsed JSON object in the reply ({} when there is none)."""
    def call(kind, sid, prompt):
        t0 = time.time()
        resp = with_retry(lambda: client.chat.completions.create(
            model=model, reasoning_effort=reasoning, messages=[{"role": "user", "content": prompt}]))
        u = resp.usage
        ec = (resp.model_extra or {}).get("estimated_cost") or {}
        try:
            cost = float(ec.get("amount"))
        except (TypeError, ValueError):
            pin, pout = PRICE.get(model, (0.25, 2.00))
            cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
        with lock:
            st["spent"] += cost
            st["calls"] += 1
            usage.write(json.dumps({"sid": sid, "kind": kind, "model": model, "in": u.prompt_tokens,
                                    "out": u.completion_tokens, "est_usd": round(cost, 6),
                                    "sec": round(time.time() - t0, 1)}) + "\n")
            usage.flush()
        m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    def ask_step(row, k, committed, is_end, note):
        return call("step", row["sid"], build_step_prompt(row, k, committed, is_end, note))

    def ask_regret(row, pieces):
        return call("regret", row["sid"], build_regret_prompt(row, pieces))
    return ask_step, ask_regret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default="models/zprobe/data/prefix14/sentences.jsonl")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default=RULES_PATH, help="rules text put in every step prompt")
    ap.add_argument("--only-sids", help="file with one sid per line: label only these")
    ap.add_argument("--notes", help="judge_schedule.py output: its failed rows' notes go into the first point's request")
    ap.add_argument("--min-gap", type=int, default=2, help="minimum tokens between two candidate points")
    ap.add_argument("--max-regret", type=int, default=2)
    a = ap.parse_args()
    RULES["path"] = a.rules

    from openai import OpenAI
    client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])

    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]
    if a.only_sids:
        only = {l.strip() for l in open(a.only_sids, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["sid"] in only]
    notes = reviewer_notes(a.notes) if a.notes else {}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    todo, spent = plan_todo(rows, a.out)
    out = open(a.out, "a", encoding="utf-8")
    fail_out = open(a.out + ".fail.jsonl", "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    lock = threading.Lock()
    st = {"spent": spent, "calls": 0, "ok": 0, "fail": 0, "error": 0, "seen": 0}
    print(f"already spent ${st['spent']:.2f} (from {a.out}.usage.jsonl); rows {len(rows)}, todo {len(todo)}", flush=True)
    ask_step, ask_regret = make_askers(client, a.model, a.reasoning, usage, lock, st)

    def work(row):
        if should_stop(st["spent"], a.budget):
            return
        reasons = []
        try:
            rec = label_sentence(row, ask_step, ask_regret, max_regret=a.max_regret, fail=reasons.append,
                                 note=notes.get(row["sid"], ""), min_gap=a.min_gap)
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
                why = reasons[-1] if reasons else "?"
                print(f"FAIL {row['sid']}: {why}", flush=True)
                fail_out.write(json.dumps({"sid": row["sid"], "reason": why}, ensure_ascii=False) + "\n")
                fail_out.flush()
            else:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                st["ok"] += 1
            if st["seen"] % 50 == 0:
                print(f"{st['seen']}/{len(todo)} ok={st['ok']} fail={st['fail']} error={st['error']} "
                      f"calls={st['calls']} est ${st['spent']:.2f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    if should_stop(st["spent"], a.budget):
        print(f"BUDGET_STOP ${st['spent']:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"LABEL_PREFIX_DONE ok={st['ok']} fail={st['fail']} error={st['error']} calls={st['calls']} "
          f"est ${st['spent']:.2f}", flush=True)
    if too_many_errors(st["error"], len(todo)):
        print(f"LABEL_PREFIX_ERRORS {st['error']} > 1% of {len(todo)} - rerun to retry them", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
