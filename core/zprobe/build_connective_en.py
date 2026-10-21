"""Connective pieces with an English target: Korean / Japanese / Chinese FLEURS train speech -> open English pieces, so the
translation adapter learns to commit English pieces mid-sentence the way segconn2 does for Korean.

Unlike build_connective.py (English source, cuts from the SEG model), the cut points come from the LLM: there is no SEG
model for Japanese or Chinese, and one rule for all three sources keeps the label style the same.

manifest : FLEURS train rows of --langs whose sentence id also has an English transcript (the gold English is the
           reference translation). One row per recording. -> {id, sid, lang, audio, src, ref}
pieces   : Qwen3-8B (thinking off) splits each sentence (once per sentence id) into 1-3 chunks and writes one English piece
           per chunk; non-final pieces stay open (no sentence-final mark), nothing from a later chunk in an earlier piece.
           Rule checks (check) and, after `hyprefix`, a leak check against Hy-MT2 translations of each chunk prefix;
           failures get one retry with the reason, the chunks kept.  -> {sid, lang, src, ref, chunks, pieces, fail}
hyprefix : Hy-MT2-1.8B translates chunk prefixes 1..i (i <= n, the last one is the whole sentence) into English for the
           leak check.
align    : Qwen3-ForcedAligner times the end of each chunk on every recording of the sentence.  -> {..., ends, dur}
build    : training rows in the segconn2 format (--seg_full): partial audio [0, end_i] -> "p1 <SEG> .. pi <SEG>", full
           audio -> "p1 <SEG> .. pn", tag "language English"; base rows from --base minus their X->en rows for these sources.

usage: python core/zprobe/build_connective_en.py manifest --out M.jsonl
       python core/zprobe/build_connective_en.py pieces --manifest M.jsonl --out P.jsonl
       PYTHONPATH=. python core/zprobe/build_connective_en.py hyprefix --pieces P.jsonl --out P_hy.jsonl
       python core/zprobe/build_connective_en.py pieces --manifest M.jsonl --hy P_hy.jsonl --out P2.jsonl   (leak check + retry)
       PYTHONPATH=Qwen3-ASR python core/zprobe/build_connective_en.py align --manifest M.jsonl --pieces P2.jsonl --out A.jsonl
       python core/zprobe/build_connective_en.py build --aligned A.jsonl --base train_segconn2.jsonl --wav_dir DIR --out T.jsonl"""
import argparse, csv, json, os, random, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FLEURS = "/home/mobility/datasets/fleurs/data"
LANG = {"ko": ("ko_kr", "Korean"), "ja": ("ja_jp", "Japanese"), "zh": ("cmn_hans_cn", "Chinese")}
NOSPACE = {"ja", "zh"}
SHORT = {"ko": "8 words", "ja": "20 characters", "zh": "20 characters"}

PROMPT = """Make English live subtitles for the {lang} sentence below, in the order it is spoken.
Split the {lang} sentence into chunks that a live subtitler would commit one after another, and write one English piece per chunk.
- Use 1 chunk if the sentence is short (under about {short}); otherwise use 2 or 3 chunks (4 only for a very long sentence). Cut only where the part so far
  can already be translated: after a clause, after a phrase ending in a particle or connective, or at a comma. Never cut inside a word.
- The chunks joined in order must give back the {lang} sentence exactly (same characters, same order, nothing left out).
- English piece i may only use what was said up to chunk i. NEVER put anything from a later chunk into an earlier piece:
  when piece i is shown, the later chunks have not been heard yet. Moving content to a later piece is fine.
- Joined with spaces, the pieces must read as one natural English sentence with the meaning of the reference translation.
  Keep the reference's word choices, names and numbers.
- Every piece except the last must stay open: it must not end with a period, question mark or exclamation mark (a comma is fine).
  The last piece ends the sentence like the reference.
Output only a JSON list of objects {{"src": "<chunk>", "en": "<piece>"}}, nothing else.

Example (Korean). Sentence: 셰이는 선거 기간 동안 마가 위기 때 나라를 떠날 수도 있다고 암시했다.
Reference: Hsieh implied during the election that Ma might flee the country during a time of crisis.
Good: [{{"src": "셰이는 선거 기간 동안", "en": "During the election, Hsieh"}}, {{"src": "마가 위기 때 나라를 떠날 수도 있다고 암시했다.", "en": "implied that Ma might flee the country during a time of crisis."}}]
Bad:  [{{"src": "셰이는 선거 기간 동안", "en": "Hsieh implied during the election"}}, ...]   (piece 1 uses "implied", which is only said in chunk 2)

{lang} sentence:
{src}

Reference translation:
{ref}"""

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that these
those there their they them he she his her we our you your i not no so than then also into about over after before such
which who whom what when where while will would can could should may might must have has had do does did""".split())


def read_tsv(lang, split):
    p = f"{FLEURS}/{LANG[lang][0] if lang in LANG else lang}/{split}.tsv"
    return [r for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE) if len(r) >= 3]


def manifest(a):
    en = {}
    for r in read_tsv("en_us", "train"):
        en.setdefault(r[0], r[2].strip())
    out = open(a.out, "w", encoding="utf-8"); n = 0
    for lang in a.langs:
        for r in read_tsv(lang, "train"):
            if r[0] not in en:
                continue
            out.write(json.dumps({"id": f"{lang}_{r[1].rsplit('.', 1)[0]}", "sid": f"{lang}_{r[0]}", "lang": lang,
                                  "audio": f"{FLEURS}/{LANG[lang][0]}/audio/train/{r[1]}", "src": r[2].strip(), "ref": en[r[0]]},
                                 ensure_ascii=False) + "\n"); n += 1
    print(f"{n} recordings")


def squash(s):
    return re.sub(r"\s+", "", s)


def words(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 3 and w not in STOP]


def resplit(chunks, src):
    """Cut the source text itself where the LLM's chunks end, counting letters and digits only, so dropped or moved
    punctuation in the LLM's copy does not matter; punctuation right after a cut stays with the earlier chunk.
    None when the LLM's chunks do not spell the source."""
    if "".join(c for c in "".join(chunks) if c.isalnum()) != "".join(c for c in src if c.isalnum()):
        return None
    marks = [alnum("".join(chunks[: k + 1])) for k in range(len(chunks) - 1)]
    out, start, cum = [], 0, 0
    for j, ch in enumerate(src):
        cum += ch.isalnum()
        if len(out) < len(marks) and cum == marks[len(out)] and ch.isalnum():
            e = j + 1
            while e < len(src) and not src[e].isalnum() and not src[e].isspace():
                e += 1
            out.append(src[start:e].strip()); start = e
    out.append(src[start:].strip())
    return out if len(out) == len(chunks) and all(out) else None


def check(chunks, pieces, sent):
    if not chunks or len(chunks) != len(pieces) or len(chunks) > 4 or not all(c.strip() and p.strip() for c, p in zip(chunks, pieces)):
        return "format"
    if resplit(chunks, sent["src"]) is None:
        return "chunks_not_source"
    small = (lambda c: len(squash(c)) < 4) if sent["lang"] in NOSPACE else (lambda c: len(c.split()) < 2)
    if len(chunks) > 1 and any(small(c) for c in chunks):
        return "tiny_chunk"
    if any(not re.search(r"[A-Za-z0-9]{2}", p) for p in pieces):
        return "empty_piece"
    for p, c in zip(pieces[:-1], chunks[:-1]):
        if re.search(r"[.!?][\"'”)]*\s*$", p.strip()) and not re.search(r"[.!?。！？][\"'”」)]*\s*$", c.strip()):
            return "closed_piece"
    if not re.search(r"[.!?][\"'”)]*\s*$", pieces[-1].strip()):
        return "open_last"
    joined = " ".join(p.strip() for p in pieces)
    ref = set(words(sent["ref"]))
    if ref and len(ref & set(words(joined))) / len(ref) < 0.4:
        return "far_from_reference"
    if len(joined) > 1.25 * len(sent["ref"]) + 10:
        return "too_long"
    toks = re.findall(r"[a-z0-9']+", joined.lower())
    grams = [tuple(toks[i: i + 4]) for i in range(len(toks) - 3)]
    if len(grams) != len(set(grams)):
        return "repeated"
    if re.search(r"[가-힣぀-ヿ一-鿿]", joined):
        return "script"
    return ""


def pulls_later(pieces, hy_prefix):
    """Content words that pieces 1..i take from a later chunk: absent from Hy-MT2's translation of chunks 1..i but present
    in its translation of the whole sentence (5-letter stems). Comparing two Hy translations, not the piece against Hy,
    keeps synonyms of the reference from counting as leaks. Two or more such words fail the row."""
    if not hy_prefix or len(hy_prefix) < 2 or not hy_prefix[-1]:
        return []
    full = {w[:5] for w in words(hy_prefix[-1])}
    for i, hp in enumerate(hy_prefix[:-1]):
        if not hp:
            continue
        have = {w[:5] for w in words(hp)}
        extra = sorted({w for w in words(" ".join(pieces[: i + 1])) if w[:5] not in have and w[:5] in full})
        if len(extra) >= 2:
            return extra
    return []


def sentences(path):
    seen, out = set(), []
    for r in map(json.loads, open(path, encoding="utf-8")):
        if r["sid"] not in seen:
            seen.add(r["sid"]); out.append({k: r[k] for k in ("sid", "lang", "src", "ref")})
    return out


def pieces(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    if a.hy:  # second pass: leak check on the first-pass labels, one retry with the chunks kept
        rows = [json.loads(l) for l in open(a.hy, encoding="utf-8")]
    else:
        rows = sentences(a.manifest)[: a.limit or None]
    rows.sort(key=lambda r: len(r["src"]))

    def msg(r, err, keep):
        c = PROMPT.format(lang=LANG[r["lang"]][1], short=SHORT[r["lang"]], src=r["src"], ref=r["ref"])
        if keep:
            c += "\n\nUse exactly these chunks: " + json.dumps(r["chunks"], ensure_ascii=False)
        if err:
            c += f"\n\nYour previous answer broke a rule ({err}). Fix it."
        return tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    def run(batch, errs, keep):
        enc = tok([msg(r, e, keep) for r, e in zip(batch, errs)], return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=500, do_sample=False, pad_token_id=tok.pad_token_id)
        res = []
        for r, seq in zip(batch, o[:, enc["input_ids"].shape[1]:]):
            txt = tok.decode(seq, skip_special_tokens=True).strip()
            m = re.search(r"\[.*\]", txt, re.S); ch = ps = None
            try:
                obj = json.loads(m.group(0)) if m else None
                ch = [str(x["src"]).strip() for x in obj]; ps = [str(x["en"]).strip() for x in obj]
            except Exception:
                pass
            why = check(ch, ps, r) if ch is not None else "format"
            if not why:
                ch = resplit(ch, r["src"])
            if not why and keep and ch != r["chunks"]:
                why = "chunks_changed"
            if not why and keep:
                extra = pulls_later(ps, r.get("hy_prefix"))
                why = f"pulls_later: {', '.join(extra)}" if extra else ""
            res.append((ch, ps, txt, why))
        return res

    out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); stats = {}
    for b in range(0, len(rows), a.batch):
        batch = rows[b: b + a.batch]
        if a.hy:
            res = []
            for r in batch:
                extra = pulls_later(r["pieces"], r.get("hy_prefix")) if not r["fail"] else []
                res.append((r["chunks"], r["pieces"], "", r["fail"] or (f"pulls_later: {', '.join(extra)}" if extra else "")))
            keep = True
        else:
            res = run(batch, [""] * len(batch), False); keep = False
        redo = [i for i, x in enumerate(res) if x[3]]
        if redo:
            again = run([batch[i] for i in redo], [res[i][3] for i in redo], keep and all(batch[i].get("chunks") for i in redo))
            for i, x in zip(redo, again):
                res[i] = x
        for r, (ch, ps, txt, why) in zip(batch, res):
            k = why.split(":")[0] or "ok"; stats[k] = stats.get(k, 0) + 1
            out.write(json.dumps({**{k_: r[k_] for k_ in ("sid", "lang", "src", "ref")}, "chunks": ch, "pieces": ps,
                                  "raw_llm": txt if why else "", "fail": why}, ensure_ascii=False) + "\n")
        out.flush()
        if (b // a.batch) % 10 == 0:
            print(f"{b + len(batch)}/{len(rows)} {time.time() - t0:.0f}s {stats}", flush=True)
    print(f"done {stats}", flush=True)


def hyprefix(a):
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
    rows = [json.loads(l) for l in open(a.pieces, encoding="utf-8")]
    jobs = []
    for i, r in enumerate(rows):
        n = len(r["chunks"] or [])
        r["hy_prefix"] = [None] * (n if n > 1 else 0)
        sep = "" if r["lang"] in NOSPACE else " "
        for k in range(len(r["hy_prefix"])):
            jobs.append((i, k, r["lang"], sep.join(r["chunks"][: k + 1])))
    jobs.sort(key=lambda j: len(j[3]))
    for b in range(0, len(jobs), 64):
        jb = jobs[b: b + 64]
        enc = tok([tr._build_prompt(t, "en", lang, None) for _, _, lang, t in jb], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for (i, k, _, _), seq in zip(jb, o[:, enc["input_ids"].shape[1]:]):
            rows[i]["hy_prefix"][k] = tok.decode(seq, skip_special_tokens=True).strip().strip('"').strip()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hy prefixes for {len(jobs)} chunk prefixes")


def alnum(s):
    return sum(1 for c in s if c.isalnum())


def align(a):
    import librosa, torch
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
    P = {r["sid"]: r for r in map(json.loads, open(a.pieces, encoding="utf-8")) if not r["fail"]}
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); n = ok = 0
    for r in map(json.loads, open(a.manifest, encoding="utf-8")):
        p = P.get(r["sid"])
        if not p:
            continue
        n += 1
        wav, _ = librosa.load(r["audio"], sr=16000); dur = len(wav) / 16000
        ends = None
        if len(p["chunks"]) == 1:
            ends = [dur]
        else:
            try:
                items = aligner.align((wav, 16000), r["src"], LANG[r["lang"]][1])[0]
                marks = [alnum("".join(p["chunks"][: k + 1])) for k in range(len(p["chunks"]) - 1)]
                cum, ends = 0, []
                for it in items:
                    cum += alnum(it.text)
                    while len(ends) < len(marks) and cum >= marks[len(ends)]:
                        ends.append(min(dur, float(it.end_time)))
                if len(ends) == len(marks) and all(x < y for x, y in zip(ends, ends[1:])) and ends[-1] < dur - 0.3:
                    ends = [round(e, 2) for e in ends] + [round(dur, 2)]
                else:
                    ends = None
            except Exception:
                ends = None
        if ends is None:
            continue
        ok += 1
        out.write(json.dumps({**r, "dur": round(dur, 2), "chunks": p["chunks"], "pieces": p["pieces"], "ends": ends},
                             ensure_ascii=False) + "\n")
        if n % 500 == 0:
            out.flush(); print(f"{n} recordings, {ok} timed, {time.time() - t0:.0f}s", flush=True)
    print(f"done {n} recordings, {ok} timed", flush=True)


def build(a):
    import soundfile as sf
    os.makedirs(a.wav_dir, exist_ok=True)
    rows = []
    for r in map(json.loads, open(a.aligned, encoding="utf-8")):
        ps, ends, pair = r["pieces"], r["ends"], f"{r['lang']}-en"
        for i in range(len(ps)):
            text = " <SEG> ".join(ps[: i + 1])
            if i < len(ps) - 1:
                cut = os.path.abspath(os.path.join(a.wav_dir, f"{r['id']}_{i + 1}.wav"))
                if not os.path.exists(cut):
                    wav, sr = sf.read(r["audio"], dtype="float32")
                    sf.write(cut, wav[: int(ends[i] * sr)], sr, subtype="PCM_16")
                rows.append({"audio": cut, "text": f"language English<asr_text>{text} <SEG>", "pair": pair})
            else:
                rows.append({"audio": r["audio"], "text": f"language English<asr_text>{text}", "pair": pair})
    drop = {f"{l}-en" for l in LANG}
    base = [{"audio": b["audio"], "text": b["text"], "pair": b["pair"]} for b in map(json.loads, open(a.base, encoding="utf-8"))
            if b["pair"] not in drop]
    allr = base + rows
    random.Random(0).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    partial = sum(r["text"].endswith("<SEG>") for r in rows)
    print(f"new X-en rows {len(rows)} ({partial} partial); base rows kept {len(base)}; total {len(allr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["manifest", "pieces", "hyprefix", "align", "build"])
    ap.add_argument("--langs", nargs="*", default=["ko", "ja", "zh"])
    ap.add_argument("--manifest"); ap.add_argument("--pieces"); ap.add_argument("--hy"); ap.add_argument("--aligned")
    ap.add_argument("--base"); ap.add_argument("--wav_dir"); ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"manifest": manifest, "pieces": pieces, "hyprefix": hyprefix, "align": align, "build": build}[a.mode](a)
