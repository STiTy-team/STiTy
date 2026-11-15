"""Register-only relabel of prefix16 schedules: Korean targets of the chosen English sources (--sources; default the
spoken ones, aihub talks and conversation) go from 합니다체 to 해요체. prefix16c passes every source: the carried
v8b/prefix13tr corpus it is mixed with is 해요체 throughout, covost and fleurs included. The piece boundaries, the empty
items (wait labels) and the wording stay: the model is asked to change only endings and polite verb forms, and a
reply that changes the number of pieces, fills an empty item or changes a word that carries no formal ending
is rejected and the record keeps its original text. No judge rerun: content does not change.

Output: --out jsonl = every input record (restyled ones get "restyled": true/false); usage per call in
<out>.usage.jsonl (cost-watch); a rerun skips sids already in --out. Exit 2 = budget stop.

usage: PYTHONPATH=. .venv/bin/python -u core/research/zprobe/labels/restyle.py \
         --schedule models/zprobe/data/prefix16/schedule.gated.jsonl --sentences models/zprobe/data/prefix14/sentences.jsonl \
         --out models/zprobe/data/prefix16/schedule.gated.restyled.jsonl"""
import argparse
import difflib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

SPOKEN_SOURCES = {"aihub", "kspon", "dailytalk"}
PRICE = {"gpt-6-luna": (0.10, 0.50)}

PROMPT = """These are the pieces of one Korean sentence, emitted one after another by a simultaneous interpreter. Rewrite them from 합니다체 into 해요체: change ONLY sentence endings and polite verb forms
(…습니다 -> …어요/아요, …입니다 -> …이에요/예요, …합니다만 -> …하지만, …습니까 -> …나요). Keep every other word, every
connective ending (…는데, …고, …며), every error and fragment exactly as it is. Keep empty pieces empty. Return a JSON
list with exactly {n} strings, one per piece, in order.

Pieces: {pieces}"""


def needs_restyle(sent: dict, sources=None) -> bool:
    return sent.get("tgt") == "ko" and sent.get("lang") == "en" and sent.get("source") in (sources or SPOKEN_SOURCES)


FORMAL = re.compile(r"(니다|니까|십시오|시오)")


def accept(old: list[str], new) -> bool:
    """Same number of pieces, empties stay empty, and inside each piece only the words that carry a formal ending
    (…니다, …니까, …십시오) may change; the word count may differ by one (…합니다만 -> …하지만 can split or merge)."""
    if not isinstance(new, list) or len(new) != len(old) or not all(isinstance(x, str) for x in new):
        return False
    for o, n in zip(old, new):
        if not o.strip():
            if n.strip():
                return False
            continue
        ow, nw = o.split(), n.split()
        if abs(len(ow) - len(nw)) > 1:
            return False
        sm = difflib.SequenceMatcher(None, ow, nw, autojunk=False)
        for tag, i1, i2, _, _ in sm.get_opcodes():
            if tag != "equal" and not all(FORMAL.search(w) for w in ow[i1:i2]):
                return False
    return True


def parse_reply(text: str):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def restyle_record(rec: dict, ask) -> dict:
    """ask(pieces) -> list of new pieces (or anything else on failure)."""
    pieces = [t for _, t in rec["schedule"]]
    new = ask(pieces)
    out = dict(rec)
    if accept(pieces, new):
        out["schedule"] = [[a, n if t.strip() else t] for (a, t), n in zip(rec["schedule"], new)]
        out["restyled"] = True
    else:
        out["restyled"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--sources", nargs="+", default=sorted(SPOKEN_SOURCES),
                    help="sentence sources whose Korean target is restyled (default: the spoken ones; pass every en source for all-해요체)")
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])
    S = {}
    for line in open(a.sentences, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            S[r["sid"]] = r
    recs = [json.loads(line) for line in open(a.schedule, encoding="utf-8") if line.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["sid"] for line in open(a.out, encoding="utf-8") if line.strip()}
    out = open(a.out, "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    lock = threading.Lock()
    spent = sum(json.loads(line)["est_usd"] for line in open(a.out + ".usage.jsonl")) if os.path.getsize(a.out + ".usage.jsonl") else 0.0
    st = {"spent": spent, "n": 0, "ok": 0, "fail": 0, "error": 0}

    def ask(pieces):
        resp = client.chat.completions.create(model=a.model, reasoning_effort=a.reasoning, messages=[
            {"role": "user", "content": PROMPT.format(n=len(pieces), pieces=json.dumps(pieces, ensure_ascii=False))}])
        u = resp.usage
        pin, pout = PRICE.get(a.model, (0.25, 2.0))
        cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
        with lock:
            st["spent"] += cost
            usage.write(json.dumps({"model": a.model, "in": u.prompt_tokens, "out": u.completion_tokens, "est_usd": round(cost, 6)}) + "\n")
            usage.flush()
        return parse_reply(resp.choices[0].message.content)

    def work(rec):
        if rec["sid"] in done:
            return
        sent = S.get(rec["sid"], {})
        if not needs_restyle(sent, set(a.sources)) or not any(t.strip() for _, t in rec["schedule"]):
            res = rec
        elif st["spent"] > a.budget:
            return
        else:
            try:
                res = restyle_record(rec, ask)
                if not res["restyled"]:
                    res = restyle_record(rec, ask)  # one retry
            except Exception as e:  # noqa: BLE001
                with lock:
                    st["error"] += 1
                    print(f"ERROR {rec['sid']}: {e}", flush=True)
                return
        with lock:
            st["n"] += 1
            if "restyled" in res:
                st["ok" if res["restyled"] else "fail"] += 1
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            if st["n"] % 1000 == 0:
                print(f"{st['n']} written, restyled ok={st['ok']} fail={st['fail']} est ${st['spent']:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, recs))
    if st["spent"] > a.budget:
        print(f"BUDGET_STOP ${st['spent']:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"RESTYLE_DONE n={st['n']} ok={st['ok']} fail={st['fail']} error={st['error']} est ${st['spent']:.3f}", flush=True)


if __name__ == "__main__":
    main()
