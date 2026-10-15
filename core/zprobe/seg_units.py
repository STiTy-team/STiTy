"""Commit-by-SEG-unit translation: the SEG fine-tuned ASR model decides the units, then each unit is translated once and
never revised. Compared with whole-sentence (offline) translation on the same FLEURS sentences.

label     : SEG model (PROBE_MODEL, no adapter) transcribes the full English audio. Its <SEG> positions are moved onto the
            gold transcript by word alignment (so ASR errors do not reach the translator — "post-corrected" units), and
            the forced aligner times the end of each unit on the gold text.
            -> {id, dur, units: [gold text ...], ends: [s ...], seg_raw: model transcript}
hy        : Hy-MT2-1.8B (bf16) translates unit i with the previous units (source + committed translation) as background,
            or with --no_context, alone.  -> {id, hyp, units_tgt, commit: [s ...]}
adapter   : translation adapter (PROBE_ADAPTER) on audio[0:end_i] with the committed translation forced as prefix; the new
            continuation is unit i's translation.  -> same fields
score     : COMET-DA of the joined units vs the FLEURS reference is done by comet_eval.py; this prints the latency side:
            units per sentence, first commit (s), and commit time per target character as a fraction of the duration.

usage: PROBE_MODEL=<seg model> PYTHONPATH=Qwen3-ASR python core/zprobe/seg_units.py label --ids A.jsonl [B.jsonl ...] --out units.jsonl
       PYTHONPATH=. python core/zprobe/seg_units.py hy --units units.jsonl --ids ev_big2_en-ko.jsonl --tgt ko --out X.jsonl
       PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/seg_units.py adapter --units units.jsonl --ids ... --tgt ko --out Y.jsonl
       python core/zprobe/seg_units.py score --files label=X.jsonl ..."""
import argparse, difflib, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
NOSPACE = {"ja", "zh"}


def continuation(tok, input_ids, seq):
    """Text generated after a forced target prefix. Decoding only the new tokens breaks a Hangul syllable whose bytes
    straddle the prefix/new boundary ("���한"), so decode prefix+new together and cut the prefix text off."""
    start = int((input_ids == tok.convert_tokens_to_ids("<asr_text>")).nonzero()[-1]) + 1
    pre = tok.decode(input_ids[start:], skip_special_tokens=False)
    full = tok.decode(seq[start:], skip_special_tokens=False)
    full = full.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return full[len(pre):] if full.startswith(pre) else full[len(pre):]


def norm(w):
    return re.sub(r"[^\w']", "", w.lower())


def project(hyp_words, gold_words, cuts):
    """Map cut positions (number of hyp words before the cut) to positions in the gold word list."""
    sm = difflib.SequenceMatcher(a=[norm(w) for w in hyp_words], b=[norm(w) for w in gold_words], autojunk=False)
    out = []
    for k in cuts:
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if i1 <= k <= i2:
                out.append(j1 + (k - i1 if tag == "equal" else round((k - i1) * (j2 - j1) / max(1, i2 - i1))))
                break
    return sorted({g for g in out if 0 < g < len(gold_words)})


def label(a):
    import librosa, torch
    import probe_z as pz
    from qwen_asr import Qwen3ForcedAligner
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
    rows = pz.read_rows("en")
    ids = sorted({json.loads(l)["id"] for f in a.ids for l in open(f, encoding="utf-8")}, key=int)
    prompt = m._build_text_prompt(context="", force_language="English")
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); nseg = 0
    for k, uid in enumerate(ids):
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / "test" / rows[uid][1]), sr=16000)
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=160)
        seq = o.sequences if hasattr(o, "sequences") else o
        raw = tok.decode(seq[0, inputs["input_ids"].shape[1]:].tolist(), skip_special_tokens=False)
        raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        parts = [p.strip() for p in raw.split("<SEG>") if p.strip()]
        hyp_words = " ".join(parts).split()
        cuts, c = [], 0
        for p in parts[:-1]:
            c += len(p.split()); cuts.append(c)
        gold = rows[uid][2].strip(); gw = gold.split()
        gcuts = project(hyp_words, gw, cuts)
        bounds = [0] + gcuts + [len(gw)]
        units = [" ".join(gw[b:e]) for b, e in zip(bounds, bounds[1:])]
        dur = len(wav) / 16000
        try:
            items = aligner.align((wav, 16000), gold, "English")[0]
            ends_w = [float(it.end_time) for it in items]
            scale = 1000.0 if ends_w and ends_w[-1] > 60 else 1.0
            ends_w = [e / scale for e in ends_w]
        except Exception:
            ends_w = []
        if len(ends_w) == len(gw):
            ends = [min(dur, ends_w[e - 1]) for e in bounds[1:-1]] + [dur]
        else:  # aligner word count differs: fall back to proportional timing
            ends = [dur * e / len(gw) for e in bounds[1:-1]] + [dur]
        nseg += len(units) - 1
        out.write(json.dumps({"id": uid, "dur": round(dur, 2), "units": units, "ends": [round(e, 2) for e in ends],
                              "seg_raw": raw, "aligned": len(ends_w) == len(gw)}, ensure_ascii=False) + "\n")
        out.flush()
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(ids)} {time.time() - t0:.0f}s, inner cuts so far {nseg}", flush=True)
    print(f"done {len(ids)} sentences, {nseg / len(ids):.2f} inner cuts per sentence", flush=True)


def join(prev, new, tgt):
    return (prev + new) if tgt in NOSPACE else (prev + " " + new).strip()


def run_hy(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.translator.local_translator import HyMTTranslator
    name = "tencent/Hy-MT2-1.8B"
    tr = HyMTTranslator(model_name=name, quant="none", context_window=0 if a.no_context else 100, always_use_context=True)
    tr._tokenizer = AutoTokenizer.from_pretrained(name); tr._tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tr._model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval(); tr._device = "cuda"
    U = {r["id"]: r for r in map(json.loads, open(a.units, encoding="utf-8"))}
    out = open(a.out, "w", encoding="utf-8")
    for r in map(json.loads, open(a.ids, encoding="utf-8")):
        u = U[r["id"]]; committed, ctx, tgt_units, commit = "", [], [], []
        for text, end in zip(u["units"], u["ends"]):
            t = time.perf_counter()
            c = None if a.no_context else ctx[-100:]
            new = tr._translate_sync(text, a.tgt, "en", c or None)
            torch.cuda.synchronize(); ms = time.perf_counter() - t
            committed = join(committed, new, a.tgt); tgt_units.append(new); commit.append(round(end + ms, 3))
            ctx.append({"text": text, "lang": "en", "translation": new})
        out.write(json.dumps({"id": r["id"], "hyp": committed, "units_tgt": tgt_units, "commit": commit, "dur": u["dur"],
                              "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"]}, ensure_ascii=False) + "\n"); out.flush()


def run_adapter(a):
    import librosa, torch
    import probe_z as pz
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    rows = pz.read_rows("en")
    U = {r["id"]: r for r in map(json.loads, open(a.units, encoding="utf-8"))}
    base = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])
    out = open(a.out, "w", encoding="utf-8")
    for r in map(json.loads, open(a.ids, encoding="utf-8")):
        u = U[r["id"]]
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / "test" / rows[r["id"]][1]), sr=16000)
        committed, tgt_units, commit = "", [], []
        for end in u["ends"]:
            t = time.perf_counter()
            if a.seg_prefix:
                prefix = "".join(t + " <SEG> " for t in tgt_units)
            else:
                prefix = committed if a.tgt in NOSPACE or not committed else committed + " "
            inputs = m.processor(text=[base + prefix], audio=[wav[: int(min(u["dur"], end + a.pad) * 16000)]], return_tensors="pt", padding=True)
            inputs = inputs.to(dev).to(torch.bfloat16)
            with torch.no_grad():
                o = m.model.generate(**inputs, max_new_tokens=120, no_repeat_ngram_size=a.no_repeat or None)
            seq = o.sequences if hasattr(o, "sequences") else o
            new = continuation(tok, inputs["input_ids"][0], seq[0])
            new = re.sub(r"\s+", " ", new.replace("<SEG>", " ")).strip()
            torch.cuda.synchronize(); ms = time.perf_counter() - t
            committed = join(committed, new, a.tgt); tgt_units.append(new)
            commit.append(round(min(u["dur"], end + a.pad) + ms, 3))
        out.write(json.dumps({"id": r["id"], "hyp": committed, "units_tgt": tgt_units, "commit": commit, "dur": u["dur"],
                              "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"]}, ensure_ascii=False) + "\n"); out.flush()


def score(a):
    print("| run | N | units / sentence | first commit (s) | commit time per char / duration |\n|---|---|---|---|---|")
    for item in a.files:
        label_, path = item.split("=", 1)
        rs = [json.loads(l) for l in open(path, encoding="utf-8")]
        nu = sum(len(r["commit"]) for r in rs) / len(rs)
        first = sum(r["commit"][0] for r in rs) / len(rs)
        frac = []
        for r in rs:
            w = [max(1, len(t)) for t in r["units_tgt"]]
            frac.append(sum(c * x for c, x in zip(r["commit"], w)) / sum(w) / r["dur"])
        print(f"| {label_} | {len(rs)} | {nu:.2f} | {first:.2f} | {sum(frac) / len(frac):.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["label", "hy", "adapter", "score"])
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--units"); ap.add_argument("--tgt"); ap.add_argument("--out")
    ap.add_argument("--no_context", action="store_true"); ap.add_argument("--no_repeat", type=int, default=0)
    ap.add_argument("--seg_prefix", action="store_true", help="adapter: force committed pieces as 'p1 <SEG> p2 <SEG> ' (--seg_full models)")
    ap.add_argument("--pad", type=float, default=0.0, help="adapter: extra audio after each unit end (s), like a SEG decision made after a short lookahead")
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    if a.mode in ("hy", "adapter"):
        a.ids = a.ids[0]
    {"label": label, "hy": run_hy, "adapter": run_adapter, "score": score}[a.mode](a)
