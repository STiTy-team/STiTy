"""Segment-wise MT with tencent/Hy-MT2-1.8B: for rows of --pair_src (e.g. en-en pseudo-labelled with <SEG>),
split the text at <SEG>, translate every segment separately into --tgt, join with ' <SEG> ', and write
rows for pair SRC-TGT (same audio). Other rows of --inp are copied unless their pair equals SRC-TGT
(those are replaced). Deterministic decoding."""
import argparse, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
LANGS = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pair_src", default="en-en"); ap.add_argument("--tgt", default="ko"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
src = a.pair_src.split("-")[0]; newpair = f"{src}-{a.tgt}"
tok = AutoTokenizer.from_pretrained("tencent/Hy-MT2-1.8B")
model = AutoModelForCausalLM.from_pretrained("tencent/Hy-MT2-1.8B", dtype=torch.bfloat16, device_map="cuda").eval()
def tr(text):
    p = f"Translate the following text into {LANGS[a.tgt]}. Note that you should only output the translated result without any additional explanation:\n\n{text}"
    ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=256, do_sample=False, repetition_penalty=1.05)
    return tok.decode(o[0][ids.shape[-1]:], skip_special_tokens=True).strip()
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
out = open(a.out, "w", encoding="utf-8"); n = 0; nseg = 0; t0 = time.time()
for r in rows:
    if r.get("pair") == newpair: continue           # replaced below
    if r.get("pair") != a.pair_src or not r.get("pseudo"):
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); continue
    out.write(json.dumps(r, ensure_ascii=False) + "\n")   # keep the ASR row too
    if a.limit and n >= a.limit: continue
    text = r["text"].split("<asr_text>", 1)[1]
    segs = [s.strip() for s in text.split("<SEG>") if s.strip()]
    trs = [tr(s) for s in segs]
    tgt_text = " <SEG> ".join(trs) + (" <SEG>" if text.rstrip().endswith("<SEG>") else "")
    out.write(json.dumps({"audio": r["audio"], "text": f"language {LANGS[a.tgt]}<asr_text>{tgt_text}", "pair": newpair, "segmt": True, "src_segs": segs}, ensure_ascii=False) + "\n")
    n += 1; nseg += len(segs)
    if n % 25 == 0: print(f"{n} utts, {nseg} segments, {time.time() - t0:.0f}s", flush=True)
    if n <= 2: print("EX:", segs, "->", trs, flush=True)
print(f"done: {n} utts, {nseg} segments translated into {a.tgt} -> {a.out}")
