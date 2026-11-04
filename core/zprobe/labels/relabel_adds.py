"""Regenerate the Korean additions (adds) of en->ko prefix labels, keeping the phrase split and its forced
alignment untouched - only the text of each addition changes, so ends[] stays valid and nothing re-aligns.

Two modes per row, picked automatically:
  gold   the row's FLEURS sentence has a human Korean translation: the model redistributes THAT translation
         into source-order additions (minimal rephrasing for connective flow) - quality anchored by gold.
  fresh  no gold (CoVoST2): the model writes the simultaneous-interpretation additions anew.

Rows to touch come from --select: "gold" (every row with a gold match) or a screen file of sids.
Every call appends usage (model, input/output tokens, est. cost) to <out>.usage.jsonl as it happens, the log
prints the running total, and --budget aborts past the cap. Validation (len(adds)==len(phrases), no
sentence-final ending before the last addition, final addition ends in haeyo style) retries once with the
reason; a row that fails twice is left out and keeps its old adds downstream.

usage: .venv/bin/python core/zprobe/labels/relabel_adds.py --select gold --limit 50 --out models/zprobe/data/relabel/pilot.jsonl"""
import argparse, csv, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

PRICE = {"gpt-5-mini": (0.25, 2.00), "gpt-5.4-mini": (0.60, 4.80),
         "gpt-6-luna": (0.10, 0.50)}  # $/M tokens (in, out); estimate only
RULES = ("Korean: natural spoken Korean a simultaneous interpreter would actually say, with connective endings "
         "(-고, -며, -는데, -면서, -지만, -어서/아서, -도록) and particles; never -다, -요, -니다 or a period before the "
         "end; end the sentence politely in 해요체 (-어요/-아요/-예요/-이에요).\n"
         "Two priorities, in this order:\n"
         "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, or "
         "content word whose English source appears only in a later phrase.\n"
         "2. Naturalness: the additions read, joined together, as one fluent Korean sentence. When emitting a "
         "phrase's content right away would force stiff phrasing or a filler tail (like '또 그렇게 해요'), leave "
         "that addition \"\" (wait) and fold the content naturally into the NEXT addition instead. Waiting is "
         "always allowed; fillers and stiffness are not.")

ap = argparse.ArgumentParser()
ap.add_argument("--aligned", default="models/zprobe/data/prefix3/aligned.jsonl")
ap.add_argument("--phrases", default="models/zprobe/data/prefix3/phrases.jsonl")
ap.add_argument("--fleurs-ko", default="/home/mobility/datasets/fleurs/data/ko_kr/train.tsv")
ap.add_argument("--select", choices=["gold", "nogold", "screen", "all"], default="all")
ap.add_argument("--screen-file", help="one sid per line (select=screen)")
ap.add_argument("--model", default="gpt-5-mini")
ap.add_argument("--reasoning", default="medium")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--budget", type=float, default=25.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # probe_z, seg_units
from build_prefix import content  # noqa: E402  (Hy-MT prefix comparison; pure python once hy is loaded)


def leaked(adds, hy):
    """build_prefix.leak with a threshold of one word: the relabel only retries on this signal (and keeps the
    old adds when the retry still fails), so the stricter gate cannot drop good rows, only flag them."""
    full = content(hy.get("full") or "", "ko")
    for i_s, hp in hy.items():
        if i_s == "full" or not hp:
            continue
        have = content(hp, "ko")
        extra = sorted(content(" ".join(adds[: int(i_s) + 1]), "ko") - have & full)
        if extra:
            return extra
    return []

from openai import OpenAI
client = OpenAI()

gold = {}
for r in csv.reader(open(a.fleurs_ko, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
    if len(r) >= 3:
        gold[r[0]] = r[2].strip()

phr = {json.loads(l)["sid"]: json.loads(l).get("phrases") or [] for l in open(a.phrases, encoding="utf-8")}
hy_path = os.path.join(os.path.dirname(a.phrases), "phrases2.jsonl")
hy = {r["sid"]: r["hy"] for l in open(hy_path, encoding="utf-8") if (r := json.loads(l)).get("hy")}

rows = []
seen = set()
for l in open(a.aligned, encoding="utf-8"):
    r = json.loads(l)
    if r["lang"] != "en" or r["tgt"] != "ko" or r["sid"] in seen:
        continue
    p = phr.get(r["sid"])
    if not p or len(p) != len(r["ends"]):
        continue
    g = gold.get(r["sid"].split("_")[1]) if r["sid"].startswith("fl_") else None
    rows.append({"sid": r["sid"], "src": r["src"], "phrases": p, "old": r["adds"], "gold": g})
    seen.add(r["sid"])

if a.select == "gold":
    rows = [r for r in rows if r["gold"]]
elif a.select == "nogold":
    rows = [r for r in rows if not r["gold"]]
elif a.select == "screen":
    want = {l.strip() for l in open(a.screen_file, encoding="utf-8")}
    rows = [r for r in rows if r["sid"] in want]
if a.limit:
    rows = rows[: a.limit]

os.makedirs(os.path.dirname(a.out), exist_ok=True)
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
out = open(a.out, "a", encoding="utf-8")
usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
spent = 0.0


def ask(row, note=""):
    global spent
    phrases = json.dumps(row["phrases"], ensure_ascii=False)
    task = "Write the simultaneous-interpretation Korean addition for each phrase."
    msg = (f"English sentence: {row['src']}\nPhrases (in order): {phrases}\n{task}\n{RULES}\n{note}\n"
           f'Answer with JSON only: {{"adds": ["...", ...]}} with exactly {len(row["phrases"])} items.')
    t0 = time.time()
    resp = client.chat.completions.create(
        model=a.model, reasoning_effort=a.reasoning,
        messages=[{"role": "user", "content": msg}])
    u = resp.usage
    pin, pout = PRICE.get(a.model, (0.25, 2.00))
    cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
    with lock:
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
        if re.search(r"(요|니다|다)\s*[.!?]\s*$", x) or re.search(r"(습니다|입니다|어요|아요|예요|이에요)$", x):
            return f"addition before the end closes the sentence: {x!r}"
    if not re.search(r"(어요|아요|예요|이에요|죠|요|해요)[.!?…]?$", body[-1]):
        return f"final addition must end in 해요체: {body[-1]!r}"
    extra = leaked(adds, hy.get(row["sid"]) or {})
    if extra:
        return ("these words are said before their English phrase has been heard (no-lookahead violation): "
                + ", ".join(extra) + " - move them to the addition of the phrase that contains them")
    return None


n_ok = n_fail = n_seen = 0
lock = threading.Lock()


def finish_sentence(adds):
    body_idx = [i for i, x in enumerate(adds) if x.strip()]
    if body_idx and not re.search(r"[.!?…]$", adds[body_idx[-1]].strip()):
        adds[body_idx[-1]] = adds[body_idx[-1]].strip() + "."
    return [x.strip() for x in adds]


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
                out.write(json.dumps({"sid": row['sid'], "mode": "fresh",
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
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(work, todo))
if spent > a.budget:
    print(f"BUDGET_STOP ${spent:.2f} > ${a.budget}", flush=True)
    sys.exit(2)
print(f"RELABEL_DONE ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)
