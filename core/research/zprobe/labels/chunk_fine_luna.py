"""Fine chunking of new x->en sentences with gpt-6-luna over the Letsur gateway, replacing the
local Qwen3-8B pieces stage for the prefix13 ko->en expansion: no GPU, better format compliance.
Prompt/validation are build_connective_en's with the --fine density (4-6 short phrases) and cap 8.
Usage appends to <out>.usage.jsonl per call; --budget aborts past the cap. Resume by sid.

usage: .venv/bin/python core/research/zprobe/labels/chunk_fine_luna.py \
  --manifest models/zprobe/data/prefix13/new_ko_manifest.jsonl \
  --out models/zprobe/data/prefix13/pieces_ko_fine.jsonl [--limit 30] [--budget 3]
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
import build_connective_en as bce  # noqa: E402

bce.FINE = True
bce.CHUNK_CAP = 8

from openai import OpenAI  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-6-luna")
ap.add_argument("--reasoning", default="low")
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget", type=float, default=3.0)
a = ap.parse_args()

client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])

seen = set()
rows = []
for l in open(a.manifest, encoding="utf-8"):
    r = json.loads(l)
    if r["sid"] in seen:
        continue
    seen.add(r["sid"])
    rows.append({k: r[k] for k in ("sid", "lang", "src", "ref")})
if a.limit:
    rows = rows[: a.limit]

done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
rows = [r for r in rows if r["sid"] not in done]
print(f"sentences: {len(rows)} (done {len(done)})", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
out = open(a.out, "a", encoding="utf-8")
usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
spent = 0.0
lock = threading.Lock()
n_ok = n_fail = n_seen = 0


def ask(r, note=""):
    global spent
    density = bce.DENSITY_FINE.format(short=bce.SHORT[r["lang"]])
    msg = bce.PROMPT.format(lang=bce.LANG[r["lang"]][1], short=bce.SHORT[r["lang"]],
                            src=r["src"], ref=r["ref"], density=density) + ("\n" + note if note else "")
    t0 = time.time()
    resp = client.chat.completions.create(model=a.model, reasoning_effort=a.reasoning,
                                          messages=[{"role": "user", "content": msg}])
    u = resp.usage
    ec = (resp.model_extra or {}).get("estimated_cost") or {}
    try:
        cost = float(ec.get("amount"))
    except (TypeError, ValueError):
        cost = u.prompt_tokens / 1e6 * 0.10 + u.completion_tokens / 1e6 * 0.50
    with lock:
        spent += cost
        usage.write(json.dumps({"sid": r["sid"], "model": a.model, "in": u.prompt_tokens,
                                "out": u.completion_tokens, "est_usd": round(cost, 6),
                                "sec": round(time.time() - t0, 1)}) + "\n")
        usage.flush()
    txt = resp.choices[0].message.content
    m = re.search(r"\[.*\]", txt, re.S)
    items = json.loads(m.group(0))
    return [i["src"] for i in items], [i["en"] for i in items], m.group(0)


def work(r):
    global n_ok, n_fail, n_seen
    if spent > a.budget:
        return
    raw = ""
    try:
        chunks, pieces, raw = ask(r)
        chunks = bce.resplit(chunks, r["src"]) or chunks
        why = bce.check(chunks, pieces, r)
        if why:
            chunks, pieces, raw = ask(r, f"Previous attempt was rejected ({why}). Fix that.")
            chunks = bce.resplit(chunks, r["src"]) or chunks
            why = bce.check(chunks, pieces, r)
    except Exception as e:  # noqa: BLE001
        why = f"error:{e}"
        chunks, pieces = None, None
    with lock:
        n_seen += 1
        if why:
            n_fail += 1
        else:
            n_ok += 1
        out.write(json.dumps({"sid": r["sid"], "lang": r["lang"], "src": r["src"], "ref": r["ref"],
                              "chunks": chunks, "pieces": pieces, "raw_llm": raw if why else "",
                              "fail": why or ""}, ensure_ascii=False) + "\n")
        out.flush()
        if n_seen % 100 == 0:
            print(f"{n_seen}/{len(rows)} ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(work, rows))
print(f"CHUNK_LUNA_DONE ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)
