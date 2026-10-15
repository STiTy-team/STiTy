"""Cascade baseline: translate source text with tencent/Hy-MT2-1.8B (bf16), on exactly the ids of a
reference jsonl (e.g. ev_big2_<pair>.jsonl) so COMET can be compared sentence by sentence.

Source text is either the base ASR transcript (--asr asr_base_<src>.jsonl, from `probe_z.py tagswap
--pair X-X` without adapter) or the gold FLEURS transcript (--gold). The prompt and the output clean-up
are the production ones (core/translator/local_translator.py HyMTTranslator); only the dtype differs
(that class loads fp16 when quant=none, this loads bf16).

Writes {id, hyp, src_text, ref_src, ref_tgt, ms, asr_ms} per line — the same fields comet_eval.py reads.
usage: PYTHONPATH=. python core/zprobe/cascade_hy.py --pair en-ko --ids ev_big2_en-ko.jsonl
         (--asr asr_base_en.jsonl | --gold) --out X.jsonl"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.translator.local_translator import HyMTTranslator

MODEL = "tencent/Hy-MT2-1.8B"

ap = argparse.ArgumentParser()
ap.add_argument("--pair", nargs="+", required=True, help="one or more SRC-TGT")
ap.add_argument("--ids", required=True, help="template with {pair}, e.g. data/ev_big2_{pair}.jsonl")
ap.add_argument("--asr", default="", help="template with {src}, e.g. data/hy/asr_base_{src}.jsonl")
ap.add_argument("--gold", action="store_true")
ap.add_argument("--out", required=True, help="template with {pair}")
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--limit", type=int, default=0, help="first N ids only (smoke)")
a = ap.parse_args()
if bool(a.asr) == a.gold:
    sys.exit("give exactly one of --asr / --gold")

tr = HyMTTranslator(model_name=MODEL, quant="none", max_new_tokens=a.max_new_tokens, context_window=0)
tr._tokenizer = AutoTokenizer.from_pretrained(MODEL)
tr._tokenizer.model_input_names = ["input_ids", "attention_mask"]  # it also returns token_type_ids, which generate() rejects
tr._model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
tr._device = "cuda"

for pair in a.pair:
    src, tgt = pair.split("-")
    refs = [json.loads(l) for l in open(a.ids.format(pair=pair), encoding="utf-8")]
    refs = refs[: a.limit] if a.limit else refs
    asr = {}
    if a.asr:
        asr = {r["id"]: r for r in map(json.loads, open(a.asr.format(src=src), encoding="utf-8"))}
        missing = [r["id"] for r in refs if r["id"] not in asr]
        if missing:
            sys.exit(f"{pair}: {len(missing)} ids have no ASR transcript, e.g. {missing[:5]}")
    out = open(a.out.format(pair=pair), "w", encoding="utf-8")
    t0 = time.time()
    for i, r in enumerate(refs):
        text = r["ref_src"] if a.gold else asr[r["id"]]["hyp"]
        t1 = time.perf_counter()
        hyp = tr._translate_sync(text, tgt, src) if text.strip() else ""
        torch.cuda.synchronize()
        ms = round((time.perf_counter() - t1) * 1000, 1)
        rec = {"id": r["id"], "hyp": hyp, "src_text": text, "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"],
               "ms": ms, "asr_ms": None if a.gold else asr[r["id"]].get("ms")}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        if (i + 1) % 50 == 0:
            print(f"[{pair} {'gold' if a.gold else 'asr'}] {i + 1}/{len(refs)}  {time.time() - t0:.0f}s", flush=True)
    out.close()
    print(f"[{pair}] done {len(refs)} in {time.time() - t0:.0f}s", flush=True)
print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
