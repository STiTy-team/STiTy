"""Distillation data from CoVoST2 English train audio: gold English sentence -> Hy-MT2-1.8B (bf16) -> ko/ja/zh target.

The teacher is the "gold transcript -> Hy" arm of the Hy comparison (COMET 0.869-0.917 on FLEURS). Each clip gets ONE
target language (by id hash), because the adapter's limit is sentence variety, not targets per sentence.

  audio : parquet shard(s) -> 16 kHz mono wav + meta.jsonl, with a length / duplicate report
  label : Hy on the gold sentences, batched -> rows {audio, text, pair, src}; drops empty, looping, off-script outputs
  merge : distilled rows + an existing train jsonl -> shuffled train jsonl

usage: python core/zprobe/build_distill_cv.py audio --parquet X.parquet [...] --out_dir DIR
       python core/zprobe/build_distill_cv.py label --out_dir DIR [--limit N]
       python core/zprobe/build_distill_cv.py merge --out_dir DIR --base train_big.jsonl --out train_big_cv.jsonl"""
import argparse, hashlib, io, json, os, random, re, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
NAMES = {"ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
OK = {"ko": lambda o: 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F,
      "zh": lambda o: 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF,
      "ja": lambda o: 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF}


def target_of(uid):
    return ["ko", "ja", "zh"][int(hashlib.md5(uid.encode()).hexdigest(), 16) % 3]


def clean_sentence(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    return re.sub(r"\s+", " ", s)


def _decode(args):
    uid, data, out = args
    import librosa, soundfile as sf
    if not os.path.exists(out):
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        y = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
        sf.write(out, y, 16000, subtype="PCM_16")
    return uid, sf_len(out)


def sf_len(p):
    import soundfile as sf
    i = sf.info(p)
    return i.frames / i.samplerate


def stage_audio(a):
    import pyarrow.parquet as pq
    wav_dir = Path(a.out_dir) / "wav"; wav_dir.mkdir(parents=True, exist_ok=True)
    meta, seen, dup = [], set(), 0
    for pf in a.parquet:
        t = pq.read_table(pf, columns=["id", "sentence", "audio"]).to_pylist()
        jobs, sent = [], {}
        for r in t:
            s = clean_sentence(r["sentence"])
            if not s:
                continue
            if s.lower() in seen:
                dup += 1; continue
            seen.add(s.lower()); sent[r["id"]] = s
            jobs.append((r["id"], r["audio"]["bytes"], str(wav_dir / f"{r['id']}.wav")))
        t0 = time.time()
        with Pool(a.workers) as pool:
            for k, (uid, dur) in enumerate(pool.imap_unordered(_decode, jobs, chunksize=32)):
                meta.append({"id": uid, "audio": str((wav_dir / f"{uid}.wav").resolve()), "src": sent[uid], "dur": round(dur, 2)})
                if (k + 1) % 5000 == 0:
                    print(f"{pf}: {k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(Path(a.out_dir) / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    d = np.array([m["dur"] for m in meta]); w = np.array([len(m["src"].split()) for m in meta])
    print(f"clips {len(meta)} (dropped {dup} duplicate sentences)")
    print(f"duration s: mean {d.mean():.1f} p50 {np.median(d):.1f} p99 {np.percentile(d, 99):.1f} max {d.max():.1f}")
    print(f"words: mean {w.mean():.1f} p50 {np.median(w):.0f} p99 {np.percentile(w, 99):.0f} max {w.max()}")
    for m in sorted(meta, key=lambda m: -m["dur"])[:3]:
        print("longest:", m["dur"], m["src"][:120])


def looping(h):
    if len(h) < 40:
        return False
    grams = [h[i:i + 8] for i in range(0, len(h) - 8)]
    return len(set(grams)) / len(grams) < 0.5


def stage_label(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.translator.local_translator import HyMTTranslator
    model_name = "tencent/Hy-MT2-1.8B"
    tr = HyMTTranslator(model_name=model_name, quant="none", context_window=0)
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tok.model_input_names = ["input_ids", "attention_mask"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr._tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda").eval()
    meta = [json.loads(l) for l in open(Path(a.out_dir) / "meta.jsonl", encoding="utf-8")]
    meta = [m for m in meta if m["dur"] <= a.max_dur]
    if a.limit:
        meta = meta[: a.limit]
    out_p = Path(a.out_dir) / "distill_rows.jsonl"
    done = set()
    if out_p.exists():
        done = {json.loads(l)["id"] for l in open(out_p, encoding="utf-8")}
    todo = sorted([m for m in meta if m["id"] not in done], key=lambda m: len(m["src"]))
    print(f"to label {len(todo)} (already {len(done)})", flush=True)
    out = open(out_p, "a", encoding="utf-8")
    drop = {"empty": 0, "loop": 0, "script": 0, "ratio": 0}
    t0 = time.time()
    for b in range(0, len(todo), a.batch):
        batch = todo[b: b + a.batch]
        prompts = [tr._build_prompt(m["src"], target_of(m["id"]), "en", None) for m in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=256, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
        for m, seq in zip(batch, o[:, enc["input_ids"].shape[1]:]):
            tgt = target_of(m["id"])
            h = tok.decode(seq, skip_special_tokens=True).strip().strip('"').strip()
            letters = [c for c in h if c.isalpha()]
            pur = sum(OK[tgt](ord(c)) for c in letters) / len(letters) if letters else 0
            ratio = len(h) / max(1, len(m["src"]))
            reason = ("empty" if not h else "loop" if looping(h) else "script" if pur < 0.9
                      else "ratio" if not 0.1 <= ratio <= 1.5 else "")
            rec = {"id": m["id"], "audio": m["audio"], "text": f"language {NAMES[tgt]}<asr_text>{h}",
                   "pair": f"en-{tgt}", "src": m["src"], "hyp": h, "drop": reason}
            if reason:
                drop[reason] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if (b // a.batch) % 50 == 0:
            n = b + len(batch)
            print(f"{n}/{len(todo)} {time.time() - t0:.0f}s ({n / max(1e-9, time.time() - t0):.1f}/s) dropped {drop}", flush=True)
    print(f"done, dropped {drop}", flush=True)


def stage_merge(a):
    rows = [json.loads(l) for l in open(Path(a.out_dir) / "distill_rows.jsonl", encoding="utf-8")]
    keep = [{"audio": r["audio"], "text": r["text"], "pair": r["pair"]} for r in rows if not r["drop"]]
    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    base = [{"audio": r["audio"], "text": r["text"], "pair": r["pair"]} for r in base]
    miss = sum(not os.path.exists(r["audio"]) for r in base)
    if miss:
        sys.exit(f"{miss} base rows point to missing audio — fix paths first (build_distill_train.py)")
    allr = base + keep
    random.Random(0).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"base {len(base)} + distilled {len(keep)} (of {len(rows)}) = {len(allr)}")
    print(dict(sorted(Counter(r["pair"] for r in allr).items())))


ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["audio", "label", "merge"])
ap.add_argument("--parquet", nargs="*", default=[]); ap.add_argument("--out_dir", required=True)
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--max_dur", type=float, default=20.0)
ap.add_argument("--base", default=""); ap.add_argument("--out", default="")
a = ap.parse_args()
{"audio": stage_audio, "label": stage_label, "merge": stage_merge}[a.stage](a)
