"""Zero-shot "transcribe, then translate" with an existing adapter, no training.

Step 1: force the SOURCE language tag and decode the transcript T (normal ASR).
Step 2: continue the same assistant text with the TARGET tag and decode the translation:
   inline : <prompt>language <Src><asr_text>T language <Tgt><asr_text>
   turn   : <prompt>language <Src><asr_text>T<|im_end|>\n<|im_start|>assistant\nlanguage <Tgt><asr_text>
Step 2 re-runs the prefill of the whole prompt (audio included), so asr_ms + trans_ms is an upper bound on the
latency; reusing step 1's KV cache would remove most of the second prefill.

Writes {id, hyp, transcript, ref_src, ref_tgt, asr_ms, trans_ms} on the ids of --ids (e.g. ev_big2_<pair>.jsonl).
usage: PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/cot_zeroshot.py --pair en-ko --ids X --variant inline --out Y"""
import argparse, json, sys, time
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_z as pz

ap = argparse.ArgumentParser()
ap.add_argument("--pair", required=True); ap.add_argument("--ids", required=True)
ap.add_argument("--variant", choices=["inline", "turn"], default="inline")
ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
src, tgt = a.pair.split("-")
m = pz.load_model()
tok = m.processor.tokenizer
dev = next(m.model.thinker.parameters()).device
rows = pz.read_rows(src)
refs = [json.loads(l) for l in open(a.ids, encoding="utf-8")]
refs = refs[: a.limit] if a.limit else refs


def run(prompt, wav):
    inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    t = time.perf_counter()
    with torch.no_grad():
        o = m.model.generate(**inputs, max_new_tokens=160)
    seq = o.sequences if hasattr(o, "sequences") else o
    out = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    torch.cuda.synchronize()
    return out, round((time.perf_counter() - t) * 1000, 1)


out = open(a.out, "w", encoding="utf-8")
t0 = time.time()
for i, r in enumerate(refs):
    wav, _ = librosa.load(str(pz.FLEURS / pz.LANGS[src][0] / "audio" / "test" / rows[r["id"]][1]), sr=16000)
    p1 = m._build_text_prompt(context="", force_language=pz.LANGS[src][1])
    transcript, asr_ms = run(p1, wav)
    tag = f"language {pz.LANGS[tgt][1]}<asr_text>"
    p2 = p1 + transcript + (f" {tag}" if a.variant == "inline" else f"<|im_end|>\n<|im_start|>assistant\n{tag}")
    hyp, trans_ms = run(p2, wav)
    rec = {"id": r["id"], "hyp": hyp, "transcript": transcript, "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"],
           "asr_ms": asr_ms, "trans_ms": trans_ms}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    if (i + 1) % 50 == 0:
        print(f"[{a.pair} {a.variant}] {i + 1}/{len(refs)} {time.time() - t0:.0f}s", flush=True)
