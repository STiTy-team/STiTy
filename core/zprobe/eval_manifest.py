"""tagswap on an arbitrary manifest instead of FLEURS: jsonl rows {id, audio, ref_src, ref_tgt}, target-language tag only.
Writes the same fields as probe_z.py tagswap ({id, hyp, ref_src, ref_tgt, ms}) so comet_eval.py / cascade_hy.py read it.
usage: PROBE_MODEL=... PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/eval_manifest.py
         --manifest X.jsonl --tgt ko --out Y.jsonl"""
import argparse, json, sys, time
from pathlib import Path

import librosa
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_z as pz

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--tgt", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
m = pz.load_model()
tok = m.processor.tokenizer
dev = next(m.model.thinker.parameters()).device
rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
out = open(a.out, "w", encoding="utf-8")
t0 = time.time()
for i, r in enumerate(rows):
    wav, _ = librosa.load(r["audio"], sr=16000)
    t1 = time.perf_counter()
    prompt = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])
    inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        o = m.model.generate(**inputs, max_new_tokens=128)
    seq = o.sequences if hasattr(o, "sequences") else o
    hyp = tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    torch.cuda.synchronize()
    rec = {"id": r["id"], "hyp": hyp, "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"],
           "ms": round((time.perf_counter() - t1) * 1000, 1)}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
    if (i + 1) % 100 == 0:
        print(f"[{a.tgt}] {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
