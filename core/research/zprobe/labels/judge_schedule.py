"""LLM judge for emission schedules (prefix15): is the joined target text a grammatical, natural, faithful
rendering of the source? COMET passes grammar slips (a dropped subject, a doubled subject, a meaning twist
scored 0.82-0.93 in the pilot); this asks a model directly. Piece boundaries are not shown - only the final
text matters - and splitting into several sentences / trailing follow-ups is explicitly allowed.

Output: --out jsonl {"sid", "defects", "note", "pass"}; usage per call in <out>.usage.jsonl
(cost-watch). A row is appended per sentence as it finishes; a rerun skips the sids already in --out and keeps
counting against --budget from the usage file. Exit 2 = budget stop, 3 = API errors on > 1% of rows.

usage: PYTHONPATH=. .venv/bin/python core/research/zprobe/labels/judge_schedule.py \
         --schedule models/zprobe/data/prefix15/pilot_v5.jsonl --sentences ... --out .../pilot_v5.judge.jsonl"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_schedule import should_stop, spent_so_far, too_many_errors  # noqa: E402
from schedule_gate import join_texts  # noqa: E402

PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-5-mini": (0.25, 2.00)}
LANG_NAME = {"en": "English", "ko": "Korean"}

PROMPT = """You are checking a simultaneous-interpretation output. Judge ONLY the final target text below.

Source ({src_lang}): {src}
Target ({tgt_lang}): {mt}{ref_line}

List the DEFECTS - things an editor would have to change for the text to be correct - using only these codes:
- grammar: a sentence is ungrammatical (missing or doubled subject, broken agreement, a dangling fragment that is not a complete sentence, a particle or ending that does not fit)
- order: word order that a native speaker would not produce (an object before its verb in English, a modifier after its noun in Korean)
- mistranslation: a word or phrase means something else than the source (e.g. "bound" rendered as "바인딩하다")
- omission: a piece of source information is missing
- addition: information not in the source was added
When a reference translation is given, a rendering that matches the reference's reading of the source is correct, not a mistranslation, omission or addition.

NOT defects (do not report these): wording that is merely awkward, literal or stiff but correct; a Korean sentence whose implied subject is rendered in English with a passive or a generic subject ("유전자를 설계해요" -> "Genes are designed."); the target being several sentences; a short follow-up sentence that adds a trailing detail ("오늘 아침에 보냈는데요."); a connective or interjection as its own sentence; spoken-style register; a transliterated name; a plausible rendering of a vague or garbled source.

Answer with JSON only: {{"defects": ["code", ...], "note": "one short sentence naming the worst problem, or empty"}}"""


def build_prompt(row, mt):
    ref_line = f"\nReference translation ({LANG_NAME[row['tgt']]}), for meaning only: {row['ref']}" if row.get("ref") else ""
    return PROMPT.format(src_lang=LANG_NAME[row["lang"]], tgt_lang=LANG_NAME[row["tgt"]], src=row["src"], mt=mt,
                         ref_line=ref_line)


CODES = ("grammar", "order", "mistranslation", "omission", "addition")


def verdict(reply):
    """Parsed judge reply -> (defects, note); malformed -> None."""
    d = reply.get("defects") if isinstance(reply, dict) else None
    if not isinstance(d, list):
        return None
    return [str(x) for x in d if str(x) in CODES], str(reply.get("note") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=10.0)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])
    S = {}
    for l in open(a.sentences, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            S[r["sid"]] = r
    recs = [json.loads(l) for l in open(a.schedule, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8") if l.strip()}
    out = open(a.out, "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    lock = threading.Lock()
    st = {"spent": spent_so_far(a.out + ".usage.jsonl"), "n": 0, "fail": 0, "error": 0}
    print(f"already spent ${st['spent']:.2f} (from {a.out}.usage.jsonl)", flush=True)

    def ask(prompt):
        t0 = time.time()
        resp = client.chat.completions.create(model=a.model, reasoning_effort=a.reasoning,
                                              messages=[{"role": "user", "content": prompt}])
        u = resp.usage
        ec = (resp.model_extra or {}).get("estimated_cost") or {}
        try:
            cost = float(ec.get("amount"))
        except (TypeError, ValueError):
            pin, pout = PRICE.get(a.model, (0.25, 2.00))
            cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
        with lock:
            st["spent"] += cost
            usage.write(json.dumps({"model": a.model, "in": u.prompt_tokens, "out": u.completion_tokens,
                                    "est_usd": round(cost, 6), "sec": round(time.time() - t0, 1)}) + "\n")
            usage.flush()
        m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}

    def work(rec):
        if should_stop(st["spent"], a.budget) or rec["sid"] not in S or not rec.get("schedule"):
            return
        row = S[rec["sid"]]
        mt = join_texts([t for _, t in rec["schedule"]])
        try:
            v = verdict(ask(build_prompt(row, mt)))
            if v is None:
                v = verdict(ask(build_prompt(row, mt)))
        except Exception as e:  # noqa: BLE001
            with lock:
                st["error"] += 1
                print(f"ERROR {rec['sid']}: {e}", flush=True)
            return
        unjudged = v is None  # two malformed replies: not a defect, the row keeps its label and is marked
        defects, note = ([], "malformed judge reply") if unjudged else v
        ok = not defects
        with lock:
            st["n"] += 1
            st["fail"] += not ok
            rec_out = {"sid": rec["sid"], "defects": defects, "note": note, "pass": ok, "mt": mt}
            if unjudged:
                rec_out["unjudged"] = True
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            out.flush()
            if st["n"] % 100 == 0:
                print(f"{st['n']} judged, fail={st['fail']} est ${st['spent']:.2f}", flush=True)

    todo = [r for r in recs if r["sid"] not in done]
    print(f"rows {len(recs)}, todo {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    if should_stop(st["spent"], a.budget):
        print(f"BUDGET_STOP ${st['spent']:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"JUDGE_DONE n={st['n']} fail={st['fail']} error={st['error']} est ${st['spent']:.3f}", flush=True)
    if too_many_errors(st["error"], len(todo)):
        print(f"JUDGE_ERRORS {st['error']} > 1% of {len(todo)} - rerun to retry them", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
