"""Regenerate the English additions (pieces) of the x->en prefix labels, keeping the chunk split and its
forced alignment untouched - the en-side twin of relabel_adds.py. The en->ko labels got the gpt-6-luna
relabel (natural spoken target, strict no-lookahead); the ko/ja/zh->en labels are still the Qwen3-8B first
pass, one label generation behind, and en is the weak direction of every delta model since.

Every row is gold-anchored: the FLEURS English reference is redistributed into source-order additions with
minimal rephrasing. Validation: exact count, non-final additions stay open (no sentence-final punctuation),
the final addition closes the sentence, and a leak check against the Hy chunk-prefix translations
(no-lookahead); one retry with the reason, a row that fails twice is left out and keeps its old pieces
downstream. Usage (tokens, est. cost) appends to <out>.usage.jsonl per call; --budget aborts past the cap.
The endpoint is the Letsur gateway (LETSUR_API_KEY).

usage: .venv/bin/python core/research/zprobe/labels/relabel_adds_en.py --limit 50 --out models/zprobe/data/relabel/pilot_en.jsonl"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-5-mini": (0.25, 2.00)}
RULES = ("English: natural spoken English a simultaneous interpreter would actually say. A reference "
         "translation of the whole sentence is given: redistribute THAT translation into source-order "
         "additions, rephrasing only as much as the source order forces; do not introduce content the "
         "reference does not have.\n"
         "Two priorities, in this order:\n"
         "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, "
         "or content word whose source appears only in a later phrase.\n"
         "2. Naturalness: the additions read, joined together, as one fluent English sentence. Additions "
         "before the last must stay syntactically open - no sentence-final period. When emitting a phrase's "
         "content right away would force stiff phrasing, leave that addition \"\" (wait) and fold the content "
         "naturally into the NEXT addition instead. Waiting is always allowed; fillers and stiffness are not.")

ap = argparse.ArgumentParser()
ap.add_argument("--pieces", default="models/zprobe/data/connective_en/pieces1.jsonl")
ap.add_argument("--hy", default="models/zprobe/data/connective_en/pieces1_hy.jsonl")
ap.add_argument("--langs", nargs="+", default=["ko", "ja", "zh"])
ap.add_argument("--model", default="gpt-6-luna")
ap.add_argument("--reasoning", default="medium")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--include-fail", action="store_true", help="also relabel rows whose first-pass pieces failed (the chunk split exists; validation gates the result)")
ap.add_argument("--budget", type=float, default=15.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_prefix import content  # noqa: E402

from openai import OpenAI  # noqa: E402
client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])

hy = {}
for l in open(a.hy, encoding="utf-8"):
    r = json.loads(l)
    hp = r.get("hy_prefix") or []
    if hp:
        hy[r["sid"]] = {**{str(i): p for i, p in enumerate(hp)}, "full": hp[-1]}

rows = []
for l in open(a.pieces, encoding="utf-8"):
    r = json.loads(l)
    if r.get("lang") not in a.langs:
        continue
    if not r.get("chunks") or not r.get("ref"):
        continue
    if r.get("fail") and not a.include_fail:
        continue
    rows.append({"sid": r["sid"], "lang": r["lang"], "src": r["src"], "ref": r["ref"],
                 "phrases": r["chunks"], "old": r.get("pieces")})
if a.limit:
    rows = rows[: a.limit]

os.makedirs(os.path.dirname(a.out), exist_ok=True)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
out = open(a.out, "a", encoding="utf-8")
usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
spent = 0.0
lock = threading.Lock()


def leaked(adds, hy_row):
    full = content(hy_row.get("full") or "", "en")
    for i_s, hp in hy_row.items():
        if i_s == "full" or not hp:
            continue
        have = content(hp, "en")
        extra = sorted(content(" ".join(adds[: int(i_s) + 1]), "en") - have & full)
        if extra:
            return extra
    return []


def ask(row, note=""):
    global spent
    phrases = json.dumps(row["phrases"], ensure_ascii=False)
    msg = (f"Source sentence ({row['lang']}): {row['src']}\n"
           f"Reference English translation: {row['ref']}\n"
           f"Phrases (in order): {phrases}\n"
           f"Write the simultaneous-interpretation English addition for each phrase.\n{RULES}\n{note}\n"
           f'Answer with JSON only: {{"adds": ["...", ...]}} with exactly {len(row["phrases"])} items.')
    t0 = time.time()
    resp = client.chat.completions.create(
        model=a.model, reasoning_effort=a.reasoning,
        messages=[{"role": "user", "content": msg}])
    u = resp.usage
    pin, pout = PRICE.get(a.model, (0.25, 2.00))
    cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
    with lock:
        global spent
        spent += cost
        usage.write(json.dumps({"sid": row["sid"], "model": a.model, "in": u.prompt_tokens,
                                "out": u.completion_tokens, "est_usd": round(cost, 6),
                                "sec": round(time.time() - t0, 1)}) + "\n")
        usage.flush()
    txt = resp.choices[0].message.content
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0))["adds"]


def check(row, adds):
    if not isinstance(adds, list) or len(adds) != len(row["phrases"]):
        return f"need exactly {len(row['phrases'])} additions"
    body = [x.strip() for x in adds if isinstance(x, str) and x.strip()]
    if not body:
        return "all additions are empty"
    for x in body[:-1]:
        if re.search(r"[.!?]$", x):
            return f"addition before the end closes the sentence: {x!r}"
    extra = leaked(adds, hy.get(row["sid"]) or {})
    if extra:
        return ("these words are said before their source phrase has been heard (no-lookahead violation): "
                + ", ".join(extra) + " - move them to the addition of the phrase that contains them")
    return None


def finish_sentence(adds):
    body_idx = [i for i, x in enumerate(adds) if x.strip()]
    if body_idx and not re.search(r"[.!?…]$", adds[body_idx[-1]].strip()):
        adds[body_idx[-1]] = adds[body_idx[-1]].strip() + "."
    return [x.strip() for x in adds]


n_ok = n_fail = n_seen = 0


def work(row):
    global n_ok, n_fail, n_seen
    if spent > a.budget:
        return
    try:
        adds = ask(row)
        why = check(row, adds)
        if why:
            adds = ask(row, f"Previous attempt was rejected: {why}. Fix that.")
            why = check(row, adds)
        with lock:
            n_seen += 1
            if why:
                n_fail += 1
                print(f"FAIL {row['sid']}: {why}", flush=True)
            else:
                out.write(json.dumps({"sid": row["sid"], "lang": row["lang"],
                                      "adds": finish_sentence(adds)}, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
            if n_seen % 50 == 0:
                print(f"{n_seen}/{len(rows)} ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)
    except Exception as e:  # noqa: BLE001
        with lock:
            n_seen += 1
            n_fail += 1
            print(f"ERROR {row['sid']}: {e}", flush=True)


todo = [r for r in rows if r["sid"] not in done]
print(f"rows {len(rows)}, todo {len(todo)}", flush=True)
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(work, todo))
if spent > a.budget:
    print(f"BUDGET_STOP ${spent:.2f} > ${a.budget}", flush=True)
    sys.exit(2)
print(f"RELABEL_EN_DONE ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)
