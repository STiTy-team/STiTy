"""Conversational-register prefix labels from corpora whose SEG cuts already exist in the SFT jsonl
(KsponSpeech ko->en, DailyTalk en->ko) - the registers FLEURS read speech never visits, and the two
target axes the delta models are short on.

The chunk split is the <SEG> cut already in the transcript, so unlike connective_en there is no
LLM chunking stage and no gold reference: gpt-6-luna writes the translation directly as source-order
additions (strict no-lookahead, non-final additions stay open, ko ends in 해요체 - the relabel_adds
rules). The leak check compares against Hy-MT2 chunk-prefix translations, same as the FLEURS pools.

manifest : parse the SFT jsonl -> {sid, lang, tgt, audio, src, chunks}; audio basenames remapped to
           --wav-dir, rows kept only if the wav exists, >= --min-chunks chunks, >= --min-words words.
hyprefix : Hy-MT2-1.8B translates chunk prefixes 1..n into --tgt (GPU; the leak reference).
adds     : luna writes one addition per chunk; validation = count / non-final open / last closes /
           leak vs hy_prefix; one retry with the reason; usage appends to <out>.usage.jsonl, --budget
           aborts past the cap. Needs LETSUR_API_KEY.
align    : Qwen3-ForcedAligner times each chunk end on the recording -> aligned rows
           {id, sid, audio, src, lang, tgt, dur, adds, ends} (the aligned_plus row shape).

usage: .venv/bin/python core/research/zprobe/labels/build_conv_pairs.py manifest \
           --src-jsonl Qwen3-ASR/finetuning/data/KSponSpeech/train_and_val.jsonl \
           --wav-dir /home/mobility/stity-datasets/ksponspeech/data/wav16k --lang ko --out M.jsonl
       PYTHONPATH=. .venv/bin/python ... hyprefix --manifest M.jsonl --out H.jsonl
       .venv/bin/python ... adds --manifest M.jsonl --hy H.jsonl --out A.jsonl [--limit 30]
       PYTHONPATH=Qwen3-ASR .venv/bin/python ... align --manifest M.jsonl --adds A.jsonl --out AL.jsonl"""
import argparse
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["manifest", "hyprefix", "adds", "align"])
ap.add_argument("--src-jsonl")
ap.add_argument("--wav-dir")
ap.add_argument("--lang", choices=["ko", "en"])
ap.add_argument("--manifest")
ap.add_argument("--hy")
ap.add_argument("--adds")
ap.add_argument("--min-chunks", type=int, default=2)
ap.add_argument("--min-words", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="gpt-6-luna")
ap.add_argument("--reasoning", default="medium")
ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--budget", type=float, default=12.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

TGT = {"ko": "en", "en": "ko"}
LANGNAME = {"ko": "Korean", "en": "English"}
PRICE = {"gpt-6-luna": (0.10, 0.50)}

RULES = {
    "en": ("English: natural spoken English a simultaneous interpreter would actually say.\n"
           "Two priorities, in this order:\n"
           "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, "
           "or content word whose source appears only in a later phrase.\n"
           "2. Naturalness: the additions read, joined together, as one fluent English sentence. Additions "
           "before the last must stay syntactically open - no sentence-final period. When emitting a phrase's "
           "content right away would force stiff phrasing, leave that addition \"\" (wait) and fold the content "
           "naturally into the NEXT addition instead. Waiting is always allowed; fillers and stiffness are not."),
    "ko": ("Korean: natural spoken Korean a simultaneous interpreter would actually say, with connective endings "
           "(-고, -며, -는데, -면서, -지만, -어서/아서, -도록) and particles; never -다, -요, -니다 or a period before the "
           "end; end the sentence politely in 해요체 (-어요/-아요/-예요/-이에요).\n"
           "Two priorities, in this order:\n"
           "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, or "
           "content word whose source appears only in a later phrase.\n"
           "2. Naturalness: the additions read, joined together, as one fluent Korean sentence. When emitting a "
           "phrase's content right away would force stiff phrasing or a filler tail, leave that addition \"\" "
           "(wait) and fold the content naturally into the NEXT addition instead. Waiting is always allowed; "
           "fillers and stiffness are not."),
}


def rows_of(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


if a.stage == "manifest":
    n_in = n_out = 0
    out = open(a.out, "w", encoding="utf-8")
    for l in open(a.src_jsonl, encoding="utf-8"):
        n_in += 1
        r = json.loads(l)
        text = r["text"].split("<asr_text>", 1)[-1]
        chunks = [c.strip() for c in text.split("<SEG>") if c.strip()]
        src = " ".join(chunks)
        wav = os.path.join(a.wav_dir, os.path.basename(r["audio"]))
        if len(chunks) < a.min_chunks or len(src.split()) < a.min_words or not os.path.exists(wav):
            continue
        sid = os.path.splitext(os.path.basename(wav))[0]
        out.write(json.dumps({"sid": sid, "id": sid, "lang": a.lang, "tgt": TGT[a.lang],
                              "audio": wav, "src": src, "chunks": chunks}, ensure_ascii=False) + "\n")
        n_out += 1
        if a.limit and n_out >= a.limit:
            break
    out.close()
    print(f"manifest: {n_out}/{n_in} rows -> {a.out}", flush=True)
    print("CONV_MANIFEST_DONE", flush=True)

elif a.stage == "hyprefix":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.translator.local_translator import HyMTTranslator
    name = "tencent/Hy-MT2-1.8B"
    tr = HyMTTranslator(model_name=name, quant="none", context_window=0)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    tok.model_input_names = ["input_ids", "attention_mask"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr._tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    rows = rows_of(a.manifest)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
    rows = [r for r in rows if r["sid"] not in done]
    jobs = []
    for i, r in enumerate(rows):
        r["hy_prefix"] = [None] * len(r["chunks"])
        for k in range(len(r["chunks"])):
            jobs.append((i, k, r["lang"], r["tgt"], " ".join(r["chunks"][: k + 1])))
    jobs.sort(key=lambda j: len(j[4]))
    t0 = time.time()
    for b in range(0, len(jobs), 64):
        jb = jobs[b: b + 64]
        enc = tok([tr._build_prompt(t, tgt, lang, None) for _, _, lang, tgt, t in jb],
                  return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
        for (i, k, _, _, _), seq in zip(jb, o):
            rows[i]["hy_prefix"][k] = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if b % 640 == 0:
            print(f"hyprefix {b}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"sid": r["sid"], "hy_prefix": r["hy_prefix"]}, ensure_ascii=False) + "\n")
    print("CONV_HYPREFIX_DONE", flush=True)

elif a.stage == "adds":
    from build_prefix import closes, content
    from openai import OpenAI
    client = OpenAI(base_url="https://gw.letsur.ai/v1", api_key=os.environ["LETSUR_API_KEY"])
    hy = {r["sid"]: r["hy_prefix"] for r in rows_of(a.hy)} if a.hy else {}
    rows = rows_of(a.manifest)
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["sid"] for l in open(a.out, encoding="utf-8")}
    out = open(a.out, "a", encoding="utf-8")
    usage = open(a.out + ".usage.jsonl", "a", encoding="utf-8")
    spent = 0.0
    lock = threading.Lock()

    def leaked(adds, hp, tgt):
        if not hp:
            return []
        full = content(hp[-1] or "", tgt)
        for i in range(len(hp) - 1):
            if not hp[i]:
                continue
            have = content(hp[i], tgt)
            extra = sorted(content(" ".join(adds[: i + 1]), tgt) - have & full)
            if extra:
                return extra
        return []

    def ask(row, note=""):
        global spent
        tgt = row["tgt"]
        phrases = json.dumps(row["chunks"], ensure_ascii=False)
        msg = (f"Source utterance ({LANGNAME[row['lang']]}, spontaneous speech): {row['src']}\n"
               f"Phrases (in order): {phrases}\n"
               f"Translate it as a simultaneous interpreter: write the {LANGNAME[tgt]} addition said "
               f"after each phrase.\n{RULES[tgt]}\n{note}\n"
               f'Answer with JSON only: {{"adds": ["...", ...]}} with exactly {len(row["chunks"])} items.')
        t0 = time.time()
        resp = client.chat.completions.create(model=a.model, reasoning_effort=a.reasoning,
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
        m = re.search(r"\{.*\}", resp.choices[0].message.content, re.S)
        return json.loads(m.group(0))["adds"]

    def check(row, adds):
        tgt = row["tgt"]
        if not isinstance(adds, list) or len(adds) != len(row["chunks"]):
            return f"need exactly {len(row['chunks'])} additions"
        if not isinstance(adds[-1], str) or not adds[-1].strip():
            return "the last addition is empty"
        body = [x.strip() for x in adds if isinstance(x, str) and x.strip()]
        for x in body[:-1]:
            if closes(x, tgt):
                return f"addition before the end closes the sentence: {x!r}"
        if not closes(body[-1], tgt):
            return f"the last addition does not close the sentence: {body[-1]!r}"
        extra = leaked(adds, hy.get(row["sid"]) or [], tgt)
        if extra:
            return ("these words are said before their source phrase has been heard (no-lookahead violation): "
                    + ", ".join(extra) + " - move them to the addition of the phrase that contains them")
        return None

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
                    out.write(json.dumps({"sid": row["sid"], "lang": row["lang"], "tgt": row["tgt"],
                                          "adds": [x.strip() for x in adds]}, ensure_ascii=False) + "\n")
                    out.flush()
                    n_ok += 1
                if n_seen % 50 == 0:
                    print(f"{n_seen}/{len(todo)} ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                n_seen += 1
                n_fail += 1
                print(f"ERROR {row['sid']}: {e}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    todo = [r for r in rows if r["sid"] not in done]
    print(f"rows {len(rows)}, todo {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    if spent > a.budget:
        print(f"BUDGET_STOP ${spent:.2f} > ${a.budget}", flush=True)
        sys.exit(2)
    print(f"CONV_ADDS_DONE ok={n_ok} fail={n_fail} est ${spent:.2f}", flush=True)

else:
    import librosa
    import torch
    from qwen_asr import Qwen3ForcedAligner

    def alnum(s):
        return sum(1 for c in s if c.isalnum())

    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B",
                                                 dtype=torch.bfloat16, device_map="cuda")
    A = {r["sid"]: r for r in rows_of(a.adds)}
    out = open(a.out, "w", encoding="utf-8")
    t0 = time.time()
    n = ok = 0
    for r in rows_of(a.manifest):
        p = A.get(r["sid"])
        if not p:
            continue
        n += 1
        wav, _ = librosa.load(r["audio"], sr=16000)
        dur = len(wav) / 16000
        ends = None
        try:
            items = aligner.align((wav, 16000), r["src"], LANGNAME[r["lang"]])[0]
            marks = [alnum("".join(r["chunks"][: k + 1])) for k in range(len(r["chunks"]) - 1)]
            cum, ends = 0, []
            for it in items:
                cum += alnum(it.text)
                while len(ends) < len(marks) and cum >= marks[len(ends)]:
                    ends.append(min(dur, float(it.end_time)))
            if len(ends) == len(marks) and all(x < y for x, y in zip(ends, ends[1:])) and ends[-1] < dur - 0.3:
                ends = [round(e, 2) for e in ends] + [round(dur, 2)]
            else:
                ends = None
        except Exception:
            ends = None
        if ends is None:
            continue
        ok += 1
        out.write(json.dumps({"id": r["id"], "sid": r["sid"], "audio": r["audio"], "src": r["src"],
                              "lang": r["lang"], "tgt": r["tgt"], "dur": round(dur, 2),
                              "adds": p["adds"], "ends": ends}, ensure_ascii=False) + "\n")
        if n % 500 == 0:
            print(f"align {n} ok={ok} {time.time() - t0:.0f}s", flush=True)
    out.close()
    print(f"align: {ok}/{n} -> {a.out}", flush=True)
    print("CONV_ALIGN_DONE", flush=True)
