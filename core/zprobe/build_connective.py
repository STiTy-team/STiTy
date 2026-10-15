"""Training labels that teach the translation adapter to commit a Korean piece that does NOT end the sentence.

The adapter closes every partial translation with a sentence-final ending ("~했다."), so in commit-and-continue the
next piece cannot attach (cut_search: en-ko loses 0.022 at SEG cuts even with the period stripped). These labels give,
for each English utterance split into SEG units, Korean pieces where every non-final piece ends with a connective
ending or particle and the joined pieces read as one 해요체 sentence.

  manifest : FLEURS en train (QUOTE_NONE) and/or CoVoST2 en clips (covost2_train_en/meta.jsonl) -> {id, audio, src}
  units    : SEG model (PROBE_MODEL) cuts on the audio, moved onto the gold text, timed by the forced aligner
  hyref    : Hy-MT2-1.8B (bf16) translates the whole sentence (lexical reference for the pieces) and every source
             prefix chunks 1..i (what can be known after chunk i; used to reject pieces that pull later content)
  pieces   : Qwen/Qwen3-8B (thinking off, bf16) splits/rewrites the Hy translation into pieces (JSON list) in 해요체;
             rule check on every piece, one retry with the broken rule named
  build    : training rows (--seg_full: pieces joined by " <SEG> " in every row). For pieces p1..pn and unit ends t1..tn:
               audio[0:t_i] -> "language Korean<asr_text>p1 .. p_i<SEG>"   (i < n)
               full audio   -> "language Korean<asr_text>p1 .. p_n"
             The en-ko rows of the base training file are dropped (their references mix registers); other rows stay.

usage: python core/zprobe/build_connective.py manifest --fleurs --covost N --out M.jsonl
       PROBE_MODEL=<seg model> PYTHONPATH=Qwen3-ASR python core/zprobe/build_connective.py units --manifest M.jsonl --out U.jsonl
       PYTHONPATH=. python core/zprobe/build_connective.py hyref --units U.jsonl --out H.jsonl
       PYTHONPATH=. python core/zprobe/build_connective.py pieces --units H.jsonl --out P.jsonl [--limit N]
       python core/zprobe/build_connective.py build --pieces P.jsonl --kiwi K.jsonl --base train.jsonl --out T.jsonl --wav_dir DIR"""
import argparse, csv, json, os, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
FLEURS_EN = Path(os.path.expanduser("~/datasets/fleurs/data/en_us"))
FINAL_END = re.compile(r"(다|요|죠|까|니다|습니다|세요|어요|아요|에요|예요)\s*$")
PUNCT_END = re.compile(r"[.!?。！？]\s*$")


def manifest(a):
    rows = []
    if a.fleurs:
        for r in csv.reader(open(FLEURS_EN / "train.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(r) >= 3 and (FLEURS_EN / "audio" / "train" / r[1]).exists():
                rows.append({"id": f"fl_{r[0]}_{Path(r[1]).stem}", "audio": str(FLEURS_EN / "audio" / "train" / r[1]),
                             "src": r[2].strip()})
    if a.covost:
        cv = [json.loads(l) for l in open("/home/mobility/datasets/covost2_train_en/meta.jsonl", encoding="utf-8")]
        cv = [m for m in cv if m["dur"] <= 20]
        for m in random.Random(0).sample(cv, min(a.covost, len(cv))):
            rows.append({"id": f"cv_{m['id']}", "audio": m["audio"], "src": m["src"]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"manifest {len(rows)} rows")


def units(a):
    import librosa, torch
    import probe_z as pz
    from seg_units import project
    from qwen_asr import Qwen3ForcedAligner
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
    prompt = m._build_text_prompt(context="", force_language="English")
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    out = open(a.out, "a", encoding="utf-8"); t0 = time.time(); n = cut = 0
    for r in map(json.loads, open(a.manifest, encoding="utf-8")):
        if r["id"] in done:
            continue
        wav, _ = librosa.load(r["audio"], sr=16000); dur = len(wav) / 16000
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=160)
        seq = o.sequences if hasattr(o, "sequences") else o
        raw = tok.decode(seq[0, inputs["input_ids"].shape[1]:].tolist(), skip_special_tokens=False)
        raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        parts = [p.strip() for p in raw.split("<SEG>") if p.strip()]
        cuts, c = [], 0
        for p in parts[:-1]:
            c += len(p.split()); cuts.append(c)
        gw = r["src"].split()
        bounds = [0] + project(" ".join(parts).split(), gw, cuts) + [len(gw)]
        ends, aligned = None, False
        if len(bounds) > 2:
            try:
                items = aligner.align((wav, 16000), r["src"], "English")[0]
                ew = [float(it.end_time) for it in items]
                scale = 1000.0 if ew and ew[-1] > 60 else 1.0
                ew = [e / scale for e in ew]
                if len(ew) == len(gw):
                    ends = [min(dur, ew[e - 1]) for e in bounds[1:-1]] + [dur]; aligned = True
            except Exception:
                pass
            if not aligned:  # cannot time the cuts reliably: keep the utterance uncut
                bounds = [0, len(gw)]
        if ends is None:
            ends = [dur]
        us = [" ".join(gw[b:e]) for b, e in zip(bounds, bounds[1:])]
        cut += len(us) - 1; n += 1
        out.write(json.dumps({**r, "dur": round(dur, 2), "units": us, "ends": [round(e, 2) for e in ends], "seg_raw": raw},
                             ensure_ascii=False) + "\n")
        if n % 200 == 0:
            out.flush(); print(f"{n} done {time.time() - t0:.0f}s, inner cuts {cut / n:.2f}/utt", flush=True)
    print(f"done {n}, inner cuts {cut / max(1, n):.2f}/utt", flush=True)


PROMPT = """Make Korean live subtitles for the English sentence below. It has been split into {n} consecutive chunks.
A reference Korean translation of the whole sentence is given: keep its word choices, names and numbers.

Write exactly {n} Korean pieces, one per chunk, in the same order:
- Piece i may only use what was said up to chunk i. NEVER put anything from a later chunk into an earlier piece:
  when piece i is shown, the later chunks have not been heard yet.
- Moving content LATER is fine and often needed in Korean: e.g. the verb of chunk i can wait for a later piece.
- Joined with spaces, the pieces must read as one natural, fluent Korean sentence.
- If a chunk does not end a sentence, its piece must NOT finish a sentence either: end it with a connective ending (e.g. -는데, -고, -며, -면서, -지만, -면, -어서/아서, -도록) or a particle (e.g. -은/는, -이/가, -을/를, -에서, -에게, -와/과), never with -다, -요, -니다 or a period.
- If a chunk ends a sentence (it ends with . ? !), its piece ends that sentence the way the reference does.
- Keep the reference's sentence endings and style; only change the endings of pieces that must stay open.
Output only a JSON list of {n} strings, nothing else.

Example. Chunks: 1. Hsieh implied during the election  2. that Ma might flee the country during a time of crisis.
Good: ["셰이는 선거 기간에", "마가 위기 때 나라를 떠날 수도 있다고 암시했어요."]   (the verb of chunk 1 waits for piece 2)
Bad:  ["셰이는 선거 기간에 마가 위기 때", "나라를 떠날 수도 있다고 암시했어요."]   (piece 1 uses "마가 위기 때" from chunk 2)

English chunks:
{chunks}

Reference translation:
{ref}"""




_KIWI = None


def haeyo(s):
    """Sentence-final endings -> polite 해요체 with kiwipiepy (했다/했습니다 -> 했어요, 됩니다 -> 돼요, -을까? -> -을까요?).
    Only final endings followed by punctuation, a closing quote or the end of the text are changed."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    s = s.strip()
    toks = _KIWI.tokenize(s)
    ms = [(t.form, t.tag) for t in toks]
    for i, t in enumerate(toks):
        nxt = ms[i + 1] if i + 1 < len(ms) else None
        if t.tag != "EF" or (nxt and nxt[1] not in ("SF", "SE", "SS", "SW")):
            continue
        if t.form.endswith("요"):
            continue
        ms[i] = (t.form + "요", "EF") if t.form.endswith("까") else ("어요", "EF")
    out = _KIWI.join(ms)
    return out.replace("아니어요", "아니에요").replace("이어요", "이에요")


def split_inside(prev, nxt):
    """True when the boundary between two pieces falls inside a Korean word ("보내" | "고 ...")."""
    haeyo("")
    a_, b_ = prev.strip().split(), nxt.strip().split()
    if not a_ or not b_ or re.search(r"[,.!?;:’”\"')]$", a_[-1]):
        return False
    w1, w2 = a_[-1], b_[0]
    pos = len(w1)
    toks = _KIWI.tokenize(w1 + w2)
    for t in toks:
        if t.start == pos:
            return t.tag[:1] in ("E", "J") or t.tag.startswith(("XS", "VCP"))
        if t.start < pos < t.start + t.len:
            return True
    return False


NOUN_TAGS = ("NNG", "NNP", "SL", "SN", "SH")


def nouns(text):
    haeyo("")
    return {t.form.lower() for t in _KIWI.tokenize(text) if t.tag in NOUN_TAGS and len(t.form) > 1}


def pulls_later(pieces, hy_prefix):
    """A piece prefix (pieces 1..i) uses a noun that the translation of chunks 1..i does not contain."""
    for i, hp in enumerate(hy_prefix):
        if not hp:
            continue
        have = nouns(hp); text = hp.lower()
        extra = [n for n in nouns(" ".join(pieces[: i + 1])) if n not in have and n not in text]
        if extra:
            return extra
    return []





def as_strings(ps):
    if isinstance(ps, list):
        return [p if isinstance(p, str) else next((v for v in p.values() if isinstance(v, str)), None) if isinstance(p, dict) else None
                for p in ps]
    return None


def check(pieces, units):
    n = len(units)
    if not isinstance(pieces, list) or len(pieces) != n or not all(isinstance(p, str) and p.strip() for p in pieces):
        return "format"
    for p, u in zip(pieces[:-1], units[:-1]):
        src_closes = bool(re.search(r'[.!?]["\u201d\')]*\s*$', u))
        closed = bool(PUNCT_END.search(p) or FINAL_END.search(p.rstrip(",")))
        if closed and not src_closes:
            return "closed_piece"
        if src_closes and not re.search(r"요[.?!]\s*$", p.strip()):
            return "register"
    if any(split_inside(p, q) for p, q in zip(pieces, pieces[1:])):
        return "split_word"
    if not re.search(r"요[.?!]?[\"\u201d']*\s*$", pieces[-1].strip()):
        return "register"
    joined = " ".join(pieces)
    if re.search(r"니다요|다요", joined):
        return "register"
    letters = [c for c in joined if c.isalpha()]
    if letters and sum(0xAC00 <= ord(c) <= 0xD7A3 for c in letters) / len(letters) < 0.8:
        return "script"
    return ""


def hyref(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.translator.local_translator import HyMTTranslator
    name = "tencent/Hy-MT2-1.8B"
    tr = HyMTTranslator(model_name=name, quant="none", context_window=0)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left"); tok.model_input_names = ["input_ids", "attention_mask"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tr._tokenizer = tok
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    rows = [json.loads(l) for l in open(a.units, encoding="utf-8")]
    jobs = [(i, -1, r["src"]) for i, r in enumerate(rows)]
    jobs += [(i, k, " ".join(r["units"][: k + 1])) for i, r in enumerate(rows) for k in range(len(r["units"]) - 1)]
    for r in rows:
        r["hy_prefix"] = [None] * (len(r["units"]) - 1)
    jobs.sort(key=lambda j: len(j[2]))
    for b in range(0, len(jobs), 64):
        jb = jobs[b: b + 64]
        enc = tok([tr._build_prompt(t, "ko", "en", None) for _, _, t in jb], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        for (i, k, _), seq in zip(jb, o[:, enc["input_ids"].shape[1]:]):
            h = tok.decode(seq, skip_special_tokens=True).strip().strip('"').strip()
            if k < 0:
                rows[i]["hy"] = h
            else:
                rows[i]["hy_prefix"][k] = h
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hy reference for {len(rows)} rows")


def pieces(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    rows = [json.loads(l) for l in open(a.units, encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    rows = sorted([r for r in rows if r["id"] not in done], key=lambda r: len(r["src"]))
    out = open(a.out, "a", encoding="utf-8"); t0 = time.time(); fails = {}

    def msg(r, err=""):
        if len(r["units"]) == 1:
            return None
        else:
            c = PROMPT.format(n=len(r["units"]), chunks="\n".join(f"{i + 1}. {u}" for i, u in enumerate(r["units"])), ref=r["hy"])
        if err:
            c += f"\n\nYour previous answer broke a rule ({err}). Fix it."
        return tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    def run(batch, errs):
        res = [None] * len(batch)
        multi = [i for i, r in enumerate(batch) if len(r["units"]) > 1]
        for i, r in enumerate(batch):
            if len(r["units"]) == 1:
                ps = [haeyo(r["hy"])]
                res[i] = (ps, "", check(ps, r["units"]))
        if not multi:
            return res
        sub = [batch[i] for i in multi]
        enc = tok([msg(r, errs[i]) for i, r in zip(multi, sub)], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, r, seq in zip(multi, sub, o[:, enc["input_ids"].shape[1]:]):
            txt = tok.decode(seq, skip_special_tokens=True).strip()
            m = re.search(r"\[.*\]", txt, re.S)
            try:
                ps = as_strings(json.loads(m.group(0))) if m else None
            except Exception:
                ps = None
            if ps and all(isinstance(x, str) for x in ps):
                ps = [haeyo(x) if PUNCT_END.search(x.strip()) or k == len(ps) - 1 else x.strip() for k, x in enumerate(ps)]
            res[i] = (ps, txt, check(ps, r["units"]))
        return res

    def verify(batch, res):
        for i, (ps, txt, why) in enumerate(res):
            if not why and len(batch[i]["units"]) > 1:
                extra = pulls_later(ps, batch[i]["hy_prefix"])
                if extra:
                    res[i] = (ps, txt + f"\n[pulls_later] {extra}", "pulls_later")
        return res

    for b in range(0, len(rows), a.batch):
        batch = rows[b: b + a.batch]
        res = verify(batch, run(batch, [""] * len(batch)))
        redo = [i for i, (_, _, why) in enumerate(res) if why and len(batch[i]["units"]) > 1]
        if redo:
            def reason(i):
                why, txt = res[i][2], res[i][1]
                if why == "pulls_later":
                    return "pieces used words from a LATER chunk: " + txt.split("[pulls_later]")[-1].strip()
                return why
            again = verify([batch[i] for i in redo], run([batch[i] for i in redo], [reason(i) for i in redo]))
            for i, rr in zip(redo, again):
                res[i] = rr
        for r, (ps, txt, why) in zip(batch, res):
            fails[why or "ok"] = fails.get(why or "ok", 0) + 1
            out.write(json.dumps({**r, "pieces": ps, "raw_llm": txt if why else "", "fail": why}, ensure_ascii=False) + "\n")
        out.flush()
        n = b + len(batch)
        if (b // a.batch) % 10 == 0:
            print(f"{n}/{len(rows)} {time.time() - t0:.0f}s {fails}", flush=True)
    print(f"done {fails}", flush=True)


def build(a):
    import soundfile as sf
    kiwi = {}
    if a.kiwi:
        rs = [json.loads(l) for l in open(a.kiwi, encoding="utf-8")]
        kiwi = {r["id"]: r["comet"] for r in rs}
    os.makedirs(a.wav_dir, exist_ok=True)
    out_rows, used, skip = [], 0, {"fail": 0, "kiwi": 0}
    for r in map(json.loads, open(a.pieces, encoding="utf-8")):
        if r["fail"]:
            skip["fail"] += 1; continue
        if kiwi and kiwi.get(r["id"], 1.0) < a.min_kiwi:
            skip["kiwi"] += 1; continue
        ps, ends = r["pieces"], r["ends"]
        used += 1
        for i in range(len(ps)):
            text = (" <SEG> " if a.seg_full else " ").join(ps[: i + 1])
            if i < len(ps) - 1:
                wav, sr = sf.read(r["audio"], dtype="float32")
                cut = os.path.abspath(os.path.join(a.wav_dir, f"{r['id']}_{i + 1}.wav"))
                if not os.path.exists(cut):
                    sf.write(cut, wav[: int(ends[i] * sr)], sr, subtype="PCM_16")
                out_rows.append({"audio": cut, "text": f"language Korean<asr_text>{text}{' <SEG>' if a.seg_full else '<SEG>'}", "pair": "en-ko"})
            else:
                out_rows.append({"audio": r["audio"], "text": f"language Korean<asr_text>{text}", "pair": "en-ko"})
    base = [json.loads(l) for l in open(a.base, encoding="utf-8")]
    base = [{"audio": b["audio"], "text": b["text"], "pair": b["pair"]} for b in base if b["pair"] != "en-ko"]
    allr = base + out_rows
    random.Random(0).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial = sum(r["text"].endswith("<SEG>") for r in out_rows)
    print(f"utterances used {used}, skipped {skip}; new en-ko rows {len(out_rows)} ({partial} partial); "
          f"base rows kept {len(base)}; total {len(allr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "units", "hyref", "pieces", "build"])
    ap.add_argument("--fleurs", action="store_true"); ap.add_argument("--covost", type=int, default=0)
    ap.add_argument("--manifest"); ap.add_argument("--units"); ap.add_argument("--pieces"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kiwi", default=""); ap.add_argument("--min_kiwi", type=float, default=0.0)
    ap.add_argument("--base"); ap.add_argument("--wav_dir")
    ap.add_argument("--seg_full", action="store_true", help="put <SEG> between pieces in every row, so the model also sees text "
                    "continuing after a <SEG> (full-audio rows) and learns to stop or continue from how much audio it has")
    a = ap.parse_args()
    {"manifest": manifest, "units": units, "hyref": hyref, "pieces": pieces, "build": build}[a.mode](a)
