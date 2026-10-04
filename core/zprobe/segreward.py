"""Stage A: future-blind segment translation under candidate segmentations, for per-target reward.
For each English FLEURS utterance: (1) SEG model ASR -> English text with <SEG> candidates,
(2) forced-align the SEG-stripped text -> boundary times, (3) for each segmentation s in
{none, all, each single boundary}: translate incrementally — audio[0:t_i] + committed target text as
forced prefix -> new segment (strip <SEG>), (4) write jsonl: utt, s, segments, concat, durations.
Scoring (COMET) is done separately in .venv (segreward_score.py)."""
import argparse, json, os, re, time
import numpy as np, librosa, torch
import sys; sys.path.insert(0, os.environ.get("PROBE_Z_DIR", os.path.dirname(os.path.abspath(__file__))))
import probe_z as P
from qwen_asr import Qwen3ForcedAligner
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test"); ap.add_argument("--tgt", default="ko"); ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--out", required=True); ap.add_argument("--max_single", type=int, default=6)
a = ap.parse_args()
m = P.load_model()                      # PROBE_MODEL=SEG model, PROBE_ADAPTER=translation adapter (merged)
tok = m.processor.tokenizer; dev = next(m.model.thinker.parameters()).device
seg_id = tok.convert_tokens_to_ids("<SEG>")
aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
rows_en, rows_t = P.read_rows("en"), P.read_rows(a.tgt)
ids = P.common_ids(["en", a.tgt])[: a.limit]
def gen(wav, force_lang, prefix=""):
    prompt = m._build_text_prompt(context="", force_language=force_lang) + prefix
    inp = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
    with torch.no_grad():
        o = m.model.generate(**inp, max_new_tokens=160)
    seq = o.sequences if hasattr(o, "sequences") else o
    g = seq[0, inp["input_ids"].shape[1]:].tolist()
    txt = tok.decode(g, skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return txt
out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); nb = 0
for n, uid in enumerate(ids):
    r = rows_en[uid]; wav, _ = librosa.load(str(P.FLEURS / "en_us" / "audio" / a.split / r[1]), sr=16000)
    dur = len(wav) / 16000
    asr = gen(wav, "English").strip()                       # English with <SEG>
    parts = [p.strip() for p in asr.split("<SEG>")]; parts = [p for p in parts if p]
    plain = " ".join(parts)
    try:
        al = aligner.align((wav, 16000), plain, "English")[0]
        items = [(it.text, float(it.start_time), float(it.end_time)) for it in al]
    except Exception as e:
        print("align fail", uid, str(e)[:80], flush=True); continue
    unit = 1000.0 if items and items[-1][2] > 60 else 1.0    # ms vs s
    # boundary times: end of the last aligned token of each part (match by cumulative word count)
    words_per_part = [len(p.split()) for p in parts]
    bounds = []; cum = 0
    for k, wpp in enumerate(words_per_part[:-1]):
        cum += wpp
        if cum - 1 < len(items): bounds.append(items[cum - 1][2] / unit)
    bounds = [b for b in bounds if 0.3 < b < dur - 0.3]
    segmentations = {"none": [], "all": bounds}
    for i, b in enumerate(bounds[: a.max_single]): segmentations[f"single{i}"] = [b]
    rec = {"id": uid, "dur": dur, "asr": asr, "bounds": bounds, "ref_src": r[2].strip(), "ref_tgt": rows_t[uid][2].strip(), "runs": {}}
    for name, bs in segmentations.items():
        cuts = sorted(bs) + [dur]; committed = ""; segs = []; prev_t = 0.0
        for t in cuts:
            piece = gen(wav[: int(t * 16000)], P.LANGS[a.tgt][1], prefix=committed)
            piece = piece.replace("<SEG>", " ").strip(); piece = re.sub(r"\s+", " ", piece)
            segs.append({"t": t, "dur": t - prev_t, "text": piece}); committed = (committed + " " + piece).strip(); prev_t = t
        rec["runs"][name] = {"bounds": bs, "segs": segs, "concat": committed, "mean_seg_dur": float(np.mean([s["dur"] for s in segs]))}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush(); nb += len(bounds)
    if n < 2: print(json.dumps({k: v["concat"] for k, v in rec["runs"].items()}, ensure_ascii=False)[:600], flush=True)
    if (n + 1) % 10 == 0: print(f"{n + 1}/{len(ids)} utts, {nb} boundaries, {time.time() - t0:.0f}s", flush=True)
print(f"done {len(ids)} utts, {nb} boundaries -> {a.out}")
