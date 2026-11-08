"""Gateway-model comparison on the two jobs we actually buy: sentence translation and prefix
additions. Reuses the conv manifests and their Hy-MT2 prefix translations, so every validity check
(format / openness / leak) runs locally and the only spend is generation.

trans : FLEURS test sentences (ko->en, en->ko), prompt = plain translation, one call per sentence.
        -> {model, dir, sid, src, ref, out}
adds  : the build_conv_pairs adds stage verbatim (same RULES, same check) on the first --limit rows
        of a conv manifest. -> {model, sid, lang, tgt, adds, fail}
Both append per-call usage to <out>.usage.jsonl and stop past --budget (per model).

usage: .venv/bin/python core/research/zprobe/labels/eval_models.py trans --models a,b --limit 5 --out T.jsonl
       .venv/bin/python core/research/zprobe/labels/eval_models.py adds --manifest M.jsonl --hy H.jsonl \
           --models a,b --limit 5 --out A.jsonl"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_conv_pairs_checks import LANGNAME, RULES, make_check  # noqa: E402
from openai import OpenAI  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["trans", "adds"])
ap.add_argument("--models", required=True, help="comma-separated gateway model ids")
ap.add_argument("--manifest")
ap.add_argument("--hy")
ap.add_argument("--fleurs", default="/home/mobility/datasets/fleurs/data")
ap.add_argument("--limit", type=int, default=5)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--reasoning", default="medium")
ap.add_argument("--budget", type=float, default=2.0, help="per-model hard stop, estimated USD")
ap.add_argument("--out", required=True)
a = ap.parse_args()

client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])
# input/output USD per 1M tokens; unknown models fall back to a deliberately high guess so the
# budget guard errs toward stopping early, and the usage log keeps raw tokens for re-pricing.
PRICE = {"gpt-6-luna": (0.10, 0.50), "gpt-5-mini": (0.25, 2.00)}
FALLBACK = (5.0, 25.0)

out = open(a.out, "a", encoding="utf-8")
usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
lock = threading.Lock()
spent = {}
done = set()
if os.path.getsize(a.out) if os.path.exists(a.out) else 0:
    done = {(r["model"], r["sid"]) for r in map(json.loads, open(a.out, encoding="utf-8"))}


def call(model, msg):
    t0 = time.time()
    kw = {"reasoning_effort": a.reasoning} if not model.startswith(("gpt-4", "gpt-5-mini")) else {}
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], **kw)
    u = resp.usage
    ec = (resp.model_extra or {}).get("estimated_cost") or {}
    try:
        cost = float(ec.get("amount"))
    except (TypeError, ValueError):
        pin, pout = PRICE.get(model, FALLBACK)
        cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
    with lock:
        spent[model] = spent.get(model, 0.0) + cost
        usage.write(json.dumps({"model": model, "in": u.prompt_tokens, "out": u.completion_tokens,
                                "est_usd": round(cost, 6), "sec": round(time.time() - t0, 1)}) + "\n")
        usage.flush()
    return resp.choices[0].message.content


def emit(row):
    with lock:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()


if a.stage == "trans":
    def read_tsv(lang):
        with open(os.path.join(a.fleurs, lang, "test.tsv"), encoding="utf-8") as f:
            return {r[0]: r[2] for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) > 2}
    ko, en = read_tsv("ko_kr"), read_tsv("en_us")
    common = sorted(set(ko) & set(en))[: a.limit]
    jobs = []
    for model in a.models.split(","):
        for sid in common:
            for d, src, ref in (("ko-en", ko[sid], en[sid]), ("en-ko", en[sid], ko[sid])):
                if (model, f"{sid}:{d}") not in done:
                    jobs.append((model, sid, d, src, ref))

    def work(j):
        model, sid, d, src, ref = j
        if spent.get(model, 0.0) > a.budget:
            return
        tgt = LANGNAME[d.split("-")[1]]
        try:
            o = call(model, f"Translate into {tgt}. Output ONLY the translation, nothing else.\n{src}")
            emit({"model": model, "sid": f"{sid}:{d}", "dir": d, "src": src, "ref": ref,
                  "out": o.strip().strip('"')})
        except Exception as e:  # noqa: BLE001
            emit({"model": model, "sid": f"{sid}:{d}", "dir": d, "error": str(e)[:200]})

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))

else:
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")][: a.limit]
    hy = {r["sid"]: r["hy_prefix"] for r in map(json.loads, open(a.hy, encoding="utf-8"))} if a.hy else {}
    check = make_check(hy)
    jobs = [(m, r) for m in a.models.split(",") for r in rows if (m, r["sid"]) not in done]

    def work(j):
        model, row = j
        if spent.get(model, 0.0) > a.budget:
            return
        tgt = row["tgt"]
        msg = (f"Source utterance ({LANGNAME[row['lang']]}, spontaneous speech): {row['src']}\n"
               f"Phrases (in order): {json.dumps(row['chunks'], ensure_ascii=False)}\n"
               f"Translate it as a simultaneous interpreter: write the {LANGNAME[tgt]} addition said "
               f"after each phrase.\n{RULES[tgt]}\n"
               f'Answer with JSON only: {{"adds": ["...", ...]}} with exactly {len(row["chunks"])} items.')
        def parse(o):
            i = o.index("{")
            return json.JSONDecoder().raw_decode(o[i:])[0]["adds"]

        try:
            try:
                adds = parse(call(model, msg))
            except Exception:  # noqa: BLE001  (parse noise is the harness's problem, not the model's first shot)
                adds = parse(call(model, msg + "\nOutput the JSON object and NOTHING else."))
            why = check(row, adds)
            emit({"model": model, "sid": row["sid"], "lang": row["lang"], "tgt": tgt,
                  "adds": adds if isinstance(adds, list) else None, "fail": why})
        except Exception as e:  # noqa: BLE001
            emit({"model": model, "sid": row["sid"], "lang": row["lang"], "tgt": tgt,
                  "adds": None, "fail": f"error: {str(e)[:160]}"})

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))

for m, s in sorted(spent.items()):
    print(f"{m} est ${s:.3f}" + (" BUDGET_STOP" if s > a.budget else ""), flush=True)
print("EVAL_STAGE_DONE", flush=True)
