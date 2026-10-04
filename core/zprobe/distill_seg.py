"""Stage C: distil a learned segmentation policy into translation targets.
For each utterance: policy (SEG model + GRPO adapter, ASR mode with the target context) -> English
transcript with <SEG>; forced aligner -> boundary times; translator (SEG model + translation adapter)
-> prefix translations per boundary; write SFT rows 'language KO<asr_text>t1 <SEG> t2 ...' (pair en-ko,
field distill=True). Then train with qwen3_asr_sft.py as in seg9.
usage: distill_seg.py --policy_adapter out_grpo_ko/adapter_step100 --trans_adapter out_segft9/final --tgt ko --split train --limit 300 --out train_distill_ko.jsonl"""
import argparse, csv, json, os, re, sys, time
import numpy as np, librosa, torch
sys.path.insert(0, os.environ.get("PROBE_Z_DIR", os.path.dirname(os.path.abspath(__file__)))); import probe_z as P
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from peft import PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--seg_model", required=True); ap.add_argument("--policy_adapter", required=True); ap.add_argument("--trans_adapter", required=True)
ap.add_argument("--tgt", default="ko"); ap.add_argument("--split", default="train"); ap.add_argument("--limit", type=int, default=300); ap.add_argument("--out", required=True)
a = ap.parse_args(); dev = "cuda"
P.MODEL = a.seg_model; P.ADAPTER = a.trans_adapter; trans = P.load_model(); ttok = trans.processor.tokenizer
pol = Qwen3ASRModel.from_pretrained(a.seg_model, dtype=torch.bfloat16, device_map="cuda", max_inference_batch_size=1)
pol.model = PeftModel.from_pretrained(pol.model, a.policy_adapter).merge_and_unload(); pol.model.eval(); ptok = pol.processor.tokenizer
aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
def read_rows(lang):
    d = {}
    for r in csv.reader(open(P.FLEURS / P.LANGS[lang][0] / f"{a.split}.tsv", encoding="utf-8"), delimiter="\t"):
        if len(r) >= 6: d.setdefault(r[0], r)
    return d
rows_en, rows_t = read_rows("en"), read_rows(a.tgt)
ids = [u for u in sorted(set(rows_en) & set(rows_t), key=int) if (P.FLEURS / "en_us" / "audio" / a.split / rows_en[u][1]).exists()][: a.limit]
ctx = f"Segment for {P.LANGS[a.tgt][1]} interpretation."
def gen(model, tok, wav, lang, prefix="", context=""):
    prompt = model._build_text_prompt(context=context, force_language=lang) + prefix
    inp = model.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad(): o = model.model.generate(**inp, max_new_tokens=160)
    seq = o.sequences if hasattr(o, "sequences") else o
    return tok.decode(seq[0, inp["input_ids"].shape[1]:].tolist(), skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); nseg = 0
for n, uid in enumerate(ids):
    r = rows_en[uid]; wp = P.FLEURS / "en_us" / "audio" / a.split / r[1]; wav, _ = librosa.load(str(wp), sr=16000); dur = len(wav) / 16000
    text = gen(pol, ptok, wav, "English", context=ctx)
    parts = [p.strip() for p in text.split("<SEG>") if p.strip()]
    bs = []
    if len(parts) > 1:
        try:
            items = [float(it.end_time) for it in aligner.align((wav, 16000), " ".join(parts), "English")[0]]
            unit = 1000.0 if items and items[-1] > 60 else 1.0; cum = 0
            for wpp in [len(p.split()) for p in parts[:-1]]:
                cum += wpp
                if cum - 1 < len(items): bs.append(items[cum - 1] / unit)
            bs = [b for b in bs if 0.3 < b < dur - 0.3]
        except Exception: bs = []
    committed = ""; segs = []
    for t in sorted(bs) + [dur]:
        piece = re.sub(r"\s+", " ", gen(trans, ttok, wav[: int(t * 16000)], P.LANGS[a.tgt][1], prefix=committed).replace("<SEG>", " ")).strip()
        segs.append(piece); committed = (committed + " " + piece).strip()
    tgt_text = " <SEG> ".join(segs) + " <SEG>"
    out.write(json.dumps({"audio": str(wp), "text": f"language {P.LANGS[a.tgt][1]}<asr_text>{tgt_text}", "pair": f"en-{a.tgt}", "distill": True, "src_seg": text, "bounds": bs}, ensure_ascii=False) + "\n"); out.flush(); nseg += len(segs)
    if (n + 1) % 25 == 0: print(f"{n + 1}/{len(ids)} utts, {nseg} segs, {time.time() - t0:.0f}s", flush=True)
print(f"done {len(ids)} utts, {nseg} segments -> {a.out}")
