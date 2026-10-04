"""Interpreter-style segment labelling with an instruct model (default unsloth/gemma-3-4b-it):
segment i is translated seeing only the PRECEDING translations as context (never the future), in a
fixed register. Rows of --pair_src (pseudo-labelled with <SEG>) become SRC-TGT rows; existing SRC-TGT rows dropped."""
import argparse, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
LANGS = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
STYLES = {"banmal": "casual 반말 as to a close friend (endings like -야, -어, -지; never -요 or -습니다)",
          "hapnida": "formal polite 합니다체 (endings -습니다 / -ㅂ니다; never -요 or 반말)",
          "haeyo": "polite spoken 해요체 (endings in -요; never -습니다 and never 반말)"}
SYS = """You are a simultaneous interpreter into {tgt}. You receive the preceding interpreted segments as context, then ONE new source segment.
Output ONLY the {tgt} interpretation of the NEW segment. The new segment may be a fragment: interpret exactly what is given, do not complete it, do not repeat or revise earlier context.
Style: {style}. Output nothing but the {tgt} text."""
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--pair_src", default="en-en"); ap.add_argument("--tgt", default="ko"); ap.add_argument("--style", default="banmal")
ap.add_argument("--model", default="unsloth/gemma-3-4b-it"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
src = a.pair_src.split("-")[0]; newpair = f"{src}-{a.tgt}"
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
sysmsg = SYS.format(tgt=LANGS[a.tgt], style=STYLES[a.style])
def interp(seg, prev):
    ctx = ("Preceding interpretation:\n" + "\n".join(prev) + "\n\n") if prev else ""
    msgs = [{"role": "user", "content": sysmsg + "\n\n" + ctx + "New segment:\n" + seg}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=120, do_sample=False)
    out = tok.decode(o[0][ids.shape[-1]:], skip_special_tokens=True).strip()
    return out.split("\n")[0].strip().strip('"')
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
out = open(a.out, "w", encoding="utf-8"); n = 0; nseg = 0; t0 = time.time()
for r in rows:
    if r.get("pair") == newpair: continue
    if r.get("pair") != a.pair_src or not r.get("pseudo"):
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); continue
    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.limit and n >= a.limit: continue
    text = r["text"].split("<asr_text>", 1)[1]
    segs = [s.strip() for s in text.split("<SEG>") if s.strip()]
    prev = []
    for s in segs: prev.append(interp(s, prev))
    tgt_text = " <SEG> ".join(prev) + (" <SEG>" if text.rstrip().endswith("<SEG>") else "")
    out.write(json.dumps({"audio": r["audio"], "text": f"language {LANGS[a.tgt]}<asr_text>{tgt_text}", "pair": newpair, "segmt": True, "style": a.style, "src_segs": segs}, ensure_ascii=False) + "\n")
    n += 1; nseg += len(segs)
    if n <= 3: print("EX:", segs, "->", prev, flush=True)
    if n % 25 == 0: print(f"{n} utts, {nseg} segs, {time.time() - t0:.0f}s", flush=True)
print(f"done: {n} utts, {nseg} segments, style={a.style} -> {a.out}")
