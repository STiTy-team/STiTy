"""Prefix-to-prefix labels: for every point in the audio, the translation the adapter may write without ever revising it,
so the output can be streamed token by token and only appended to.

An LLM writes a simultaneous-interpretation translation in source order: the source is split into short phrases and for
each phrase the text an interpreter adds right after hearing it (possibly nothing, when it must wait). The forced aligner
times every phrase end on each recording. A training row is then audio[0:t] -> everything added up to the last phrase that
ended before t, closed by EOS; only the full audio gets the sentence-final ending. No <SEG> anywhere: the model learns to
stop where the audio stops being enough.

Sources: English speech -> Korean (data/connective: FLEURS + CoVoST2 English, Hy-MT2 Korean as reference) and
Korean/Japanese/Chinese speech -> English (data/connective_en: FLEURS train, gold English as reference).

sents    : one row per sentence from both manifests.  -> {sid, lang, tgt, src, ref}
phrases  : Qwen3-8B on vLLM (thinking off) writes [{src, add}]; rule checks, one retry with the reason.
hyprefix : Hy-MT2 translates phrase prefixes that end with a non-empty addition, plus the whole sentence (leak check).
leak     : drops rows whose additions up to phrase i use content words that Hy only has once later phrases are added;
           one retry with the phrases kept.
align    : phrase end times on every recording (Qwen3-ForcedAligner).
build    : boundary rows (up to --per_rec), one mid-phrase row and the full row per recording, plus base rows from --base
           that neither target Korean/English from another language nor carry <SEG>.

usage (see chains/prefix.sh): sents -> phrases -> hyprefix -> leak -> align -> build"""
import argparse, json, os, random, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_connective as bc  # noqa: E402  (haeyo, split_inside, nouns, FINAL_END, PUNCT_END)
import build_connective_en as be  # noqa: E402  (resplit, words, alnum)

NAME = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}
NOSPACE = {"ja", "zh"}
UNIT = {"en": "2 to 5 words", "ko": "1 to 3 words", "ja": "3 to 8 characters", "zh": "3 to 8 characters"}
RULES = {
    "ko": "Korean: follow the source order with connective endings (-고, -며, -는데, -면서, -지만, -어서/아서, -도록) and particles, "
          "never with -다, -요, -니다 or a period before the end; end the sentence politely in 해요체 (-어요/-아요/-예요/-이에요).",
    "en": "English: one sentence that grows in source order; no period, question or exclamation mark before the end.",
}
EXAMPLE = {
    "ko": """Example. Sentence: Hsieh implied during the election that Ma might flee the country during a time of crisis.
Reference: 셰이는 선거 기간에 마가 위기 때 나라를 떠날 수도 있다고 암시했어요.
Good: [{"src": "Hsieh implied", "add": "셰이는"}, {"src": "during the election", "add": "선거 기간에"}, {"src": "that Ma might flee", "add": "마가"},
       {"src": "the country", "add": "나라를 떠날 수도 있다고"}, {"src": "during a time of crisis.", "add": "위기 때 그럴 수 있다고 암시했어요."}]
Bad:  [{"src": "Hsieh implied", "add": "셰이는 암시했어요"}, ...]   (closes the sentence too early)
Bad:  [{"src": "Hsieh implied", "add": "셰이는 선거 기간에"}, ...]   (uses "during the election" before it is heard)""",
    "en": """Example (Korean). Sentence: 셰이는 선거 기간 동안 마가 위기 때 나라를 떠날 수도 있다고 암시했다.
Reference: Hsieh implied during the election that Ma might flee the country during a time of crisis.
Good: [{"src": "셰이는", "add": "Hsieh,"}, {"src": "선거 기간 동안", "add": "during the election,"}, {"src": "마가 위기 때", "add": "said that Ma, in a time of crisis,"},
       {"src": "나라를 떠날 수도 있다고", "add": "might flee the country"}, {"src": "암시했다.", "add": "— or so he implied."}]
Bad:  [{"src": "셰이는", "add": "Hsieh implied"}, ...]   (uses "implied" before it is heard)""",
}
PROMPT = """Make {tgt} simultaneous-interpretation subtitles for the {src_lang} sentence below. An interpreter translates while
listening: follow the source order and do not wait for the end of the sentence unless you must.
Split the {src_lang} sentence into consecutive short phrases of about {unit} (never cut inside a word). For each phrase,
write the {tgt} text the interpreter adds right after hearing it.
- An addition may only use what was said up to its phrase. NEVER use anything from a later phrase: it has not been heard.
  If a phrase cannot be translated yet, its addition may be "" and the words come later.
- The phrases joined in order must give back the {src_lang} sentence exactly.
- All additions joined with spaces must read as one fluent {tgt} sentence with the meaning of the reference translation;
  keep its names and numbers.
- {rules}
- Only the last addition ends the sentence, and it must not be "".
Output only a JSON list of objects {{"src": "<phrase>", "add": "<{tgt} text>"}}, nothing else.

{example}

{src_lang} sentence:
{src}

Reference translation:
{ref}"""


def sents(a):
    out, seen = [], set()
    hy = {r["id"]: r.get("hy") for r in map(json.loads, open(a.ko_ref, encoding="utf-8"))}
    for r in map(json.loads, open(a.en_manifest, encoding="utf-8")):  # English speech -> Korean
        if hy.get(r["id"]) and r["id"] not in seen:
            seen.add(r["id"]); out.append({"sid": r["id"], "lang": "en", "tgt": "ko", "src": r["src"], "ref": hy[r["id"]]})
    for r in map(json.loads, open(a.x_manifest, encoding="utf-8")):  # ko/ja/zh speech -> English
        if r["sid"] not in seen:
            seen.add(r["sid"]); out.append({"sid": r["sid"], "lang": r["lang"], "tgt": "en", "src": r["src"], "ref": r["ref"]})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} sentences: {sum(r['tgt'] == 'ko' for r in out)} -> ko, {sum(r['tgt'] == 'en' for r in out)} -> en")


def closes(text, tgt):
    t = text.strip().rstrip("\"'”’)")
    return bool(bc.PUNCT_END.search(t) or (tgt == "ko" and bc.FINAL_END.search(t.rstrip(",").rstrip())))


def content(text, tgt):
    return {w[:2] if tgt == "ko" else w[:5] for w in (bc.nouns(text) if tgt == "ko" else be.words(text))}


def check(ph, adds, r):
    tgt = r["tgt"]
    if not ph or len(ph) != len(adds) or len(ph) > 14 or not all(p.strip() for p in ph):
        return "format"
    if be.resplit(ph, r["src"]) is None:
        return "phrases_not_source"
    if not adds[-1].strip():
        return "empty_last"
    for i, (p, x) in enumerate(zip(ph[:-1], adds[:-1])):
        if x.strip() and closes(x, tgt) and not re.search(r"[.!?。！？][\"'”」)]*\s*$", p.strip()):
            return "closed_early"
    if not closes(adds[-1], tgt):
        return "open_last"
    nonempty = [x for x in adds if x.strip()]
    if tgt == "ko":
        if any(bc.split_inside(x, y) for x, y in zip(nonempty, nonempty[1:])):
            return "split_word"
        if not re.search(r"요[.?!]?[\"”']*\s*$", adds[-1].strip()):
            return "register"
    joined = " ".join(x.strip() for x in nonempty)
    if len(joined) > 1.3 * len(r["ref"]) + 10:
        return "too_long"
    toks = re.findall(r"\w+", joined.lower())
    grams = [tuple(toks[i: i + 4]) for i in range(len(toks) - 3)]
    if len(grams) != len(set(grams)):
        return "repeated"
    ref = content(r["ref"], tgt)
    if ref and len(ref & content(joined, tgt)) / len(ref) < 0.4:
        return "far_from_reference"
    letters = [c for c in joined if c.isalpha()]
    if tgt == "ko" and letters and sum(0xAC00 <= ord(c) <= 0xD7A3 for c in letters) / len(letters) < 0.8:
        return "script"
    if tgt == "en" and re.search(r"[가-힣぀-ヿ一-鿿]", joined):
        return "script"
    return ""


def leak(adds, hy, tgt):
    """Content words in additions 1..i that Hy-MT2 only produces once later phrases are added (2+ fail)."""
    full = content(hy.get("full") or "", tgt)
    for i_s, hp in hy.items():
        if i_s == "full" or not hp:
            continue
        i = int(i_s)
        have = content(hp, tgt)
        extra = sorted(content(" ".join(adds[: i + 1]), tgt) - have & full)
        if len(extra) >= 2:
            return extra
    return []


def phrases(a):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    name = "Qwen/Qwen3-8B"
    tok = AutoTokenizer.from_pretrained(name)
    llm = LLM(model=name, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=900)
    if a.hy:
        rows = [json.loads(l) for l in open(a.hy, encoding="utf-8")]
    else:
        rows = [json.loads(l) for l in open(a.sents, encoding="utf-8")][: a.limit or None]

    def msg(r, err, keep):
        c = PROMPT.format(tgt=NAME[r["tgt"]], src_lang=NAME[r["lang"]], unit=UNIT[r["lang"]], rules=RULES[r["tgt"]],
                          example=EXAMPLE[r["tgt"]], src=r["src"], ref=r["ref"])
        if keep:
            c += "\n\nUse exactly these phrases: " + json.dumps(r["phrases"], ensure_ascii=False)
        if err:
            c += f"\n\nYour previous answer broke a rule ({err}). Fix it."
        return tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    def parse(r, txt, keep):
        m = re.search(r"\[.*\]", txt, re.S); ph = adds = None
        try:
            obj = json.loads(m.group(0)) if m else None
            ph = [str(x["src"]).strip() for x in obj]; adds = [str(x.get("add", "")).strip() for x in obj]
        except Exception:
            return None, None, "format"
        if r["tgt"] == "ko" and adds and adds[-1]:
            adds[-1] = bc.haeyo(adds[-1])
        why = check(ph, adds, r)
        if not why:
            ph = be.resplit(ph, r["src"])
            if keep and ph != r["phrases"]:
                why = "phrases_changed"
        if not why and keep:
            extra = leak(adds, r.get("hy") or {}, r["tgt"])
            why = f"uses later words: {', '.join(extra)}" if extra else ""
        return ph, adds, why

    def run(batch, errs, keep):
        outs = llm.generate([msg(r, e, keep) for r, e in zip(batch, errs)], sp, use_tqdm=False)
        return [(*parse(r, o.outputs[0].text, keep), o.outputs[0].text) for r, o in zip(batch, outs)]

    t0 = time.time(); stats = {}
    out = open(a.out, "w", encoding="utf-8")
    for b in range(0, len(rows), a.chunk):
        batch = rows[b: b + a.chunk]
        if a.hy:
            res, keep = [], True
            for r in batch:
                extra = leak(r["adds"], r.get("hy") or {}, r["tgt"]) if not r["fail"] else []
                res.append((r["phrases"], r["adds"], r["fail"] or (f"uses later words: {', '.join(extra)}" if extra else ""), ""))
        else:
            res, keep = run(batch, [""] * len(batch), False), False
        redo = [i for i, x in enumerate(res) if x[2]]
        if redo:
            sub = [batch[i] for i in redo]
            again = run(sub, [res[i][2] for i in redo], keep and all(r.get("phrases") for r in sub))
            for i, x in zip(redo, again):
                res[i] = x
        for r, (ph, adds, why, txt) in zip(batch, res):
            k = why.split(":")[0] or "ok"; stats[k] = stats.get(k, 0) + 1
            out.write(json.dumps({**{k_: r[k_] for k_ in ("sid", "lang", "tgt", "src", "ref")}, "phrases": ph, "adds": adds,
                                  "raw_llm": txt if why else "", "fail": why}, ensure_ascii=False) + "\n")
        out.flush()
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
    rows = [json.loads(l) for l in open(a.phrases, encoding="utf-8")]
    jobs = []
    for n, r in enumerate(rows):
        r["hy"] = {}
        if r["fail"]:
            continue
        sep = "" if r["lang"] in NOSPACE else " "
        ks = [i for i, x in enumerate(r["adds"][:-1]) if x.strip()]
        for i in ks:
            jobs.append((n, str(i), r["lang"], r["tgt"], sep.join(r["phrases"][: i + 1])))
        if ks:
            jobs.append((n, "full", r["lang"], r["tgt"], r["src"]))
    jobs.sort(key=lambda j: len(j[4]))
    t0 = time.time()
    for b in range(0, len(jobs), 96):
        jb = jobs[b: b + 96]
        enc = tok([tr._build_prompt(t, tgt, lang, None) for _, _, lang, tgt, t in jb], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        for (n, k, _, _, _), seq in zip(jb, o[:, enc["input_ids"].shape[1]:]):
            rows[n]["hy"][k] = tok.decode(seq, skip_special_tokens=True).strip().strip('"').strip()
        if (b // 96) % 100 == 0:
            print(f"{b + len(jb)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"hy prefixes {len(jobs)}")


def align(a):
    import librosa, torch
    from qwen_asr import Qwen3ForcedAligner
    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
    P = {r["sid"]: r for r in map(json.loads, open(a.phrases, encoding="utf-8")) if not r["fail"]}
    recs = []
    for r in map(json.loads, open(a.en_manifest, encoding="utf-8")):
        recs.append({"id": r["id"], "sid": r["id"], "audio": r["audio"], "src": r["src"]})
    for r in map(json.loads, open(a.x_manifest, encoding="utf-8")):
        recs.append({"id": r["id"], "sid": r["sid"], "audio": r["audio"], "src": r["src"]})
    rng = random.Random(0)
    by_lang = {}
    for r in recs:
        if r["sid"] in P:
            by_lang.setdefault(P[r["sid"]]["lang"], []).append(r)
    if a.max_en:
        rng.shuffle(by_lang.get("en", []))
        fl = [r for r in by_lang.get("en", []) if r["id"].startswith("fl_")]
        cv = [r for r in by_lang.get("en", []) if not r["id"].startswith("fl_")]
        by_lang["en"] = fl + cv[: max(0, a.max_en - len(fl))]
    recs = [r for v in by_lang.values() for r in v]
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); ok = 0
    for n, r in enumerate(recs):
        p = P[r["sid"]]
        wav, _ = librosa.load(r["audio"], sr=16000); dur = len(wav) / 16000
        try:
            items = aligner.align((wav, 16000), r["src"], NAME[p["lang"]])[0]
        except Exception:
            continue
        marks = [be.alnum("".join(p["phrases"][: k + 1])) for k in range(len(p["phrases"]) - 1)]
        cum, ends = 0, []
        for it in items:
            cum += be.alnum(it.text)
            while len(ends) < len(marks) and cum >= marks[len(ends)]:
                ends.append(min(dur, float(it.end_time)))
        if len(ends) != len(marks) or any(y < x for x, y in zip(ends, ends[1:])) or (ends and ends[-1] > dur - 0.2):
            continue
        ok += 1
        out.write(json.dumps({**r, "lang": p["lang"], "tgt": p["tgt"], "dur": round(dur, 2), "adds": p["adds"],
                              "ends": [round(e, 2) for e in ends] + [round(dur, 2)]}, ensure_ascii=False) + "\n")
        if (n + 1) % 1000 == 0:
            out.flush(); print(f"{n + 1}/{len(recs)} {time.time() - t0:.0f}s, timed {ok}", flush=True)
    print(f"done {len(recs)} recordings, timed {ok}", flush=True)


def build(a):
    import soundfile as sf
    os.makedirs(a.wav_dir, exist_ok=True)
    rng = random.Random(0); rows = []; kinds = {"boundary": 0, "mid": 0, "full": 0}
    for r in map(json.loads, open(a.aligned, encoding="utf-8")):
        adds, ends, tgt = r["adds"], r["ends"], r["tgt"]
        tag = f"language {NAME[tgt]}<asr_text>"
        cum = [" ".join(x.strip() for x in adds[: i + 1] if x.strip()) for i in range(len(adds))]
        wav = sr = None

        def cut(t, name):
            nonlocal wav, sr
            path = os.path.abspath(os.path.join(a.wav_dir, f"{r['id']}_{name}.wav"))
            if not os.path.exists(path):
                if wav is None:
                    wav, sr = sf.read(r["audio"], dtype="float32")
                sf.write(path, wav[: int(t * sr)], sr, subtype="PCM_16")
            return path
        bounds = [i for i in range(len(adds) - 1) if ends[i] >= 0.6]
        for i in sorted(rng.sample(bounds, min(a.per_rec, len(bounds)))):
            rows.append({"audio": cut(ends[i] + a.pad, f"b{i}"), "text": tag + cum[i], "pair": f"{r['lang']}-{tgt}"}); kinds["boundary"] += 1
        spans = [(j, (ends[j - 1] if j else 0.0) + 0.15, ends[j] - 0.15) for j in range(len(adds))]
        spans = [(j, lo, hi) for j, lo, hi in spans if hi - lo >= 0.3 and hi >= 0.8]
        if spans:
            j, lo, hi = rng.choice(spans)
            rows.append({"audio": cut(rng.uniform(max(lo, 0.6), hi), f"m{j}"), "text": tag + (cum[j - 1] if j else ""),
                         "pair": f"{r['lang']}-{tgt}"}); kinds["mid"] += 1
        rows.append({"audio": r["audio"], "text": tag + cum[-1], "pair": f"{r['lang']}-{tgt}"}); kinds["full"] += 1
    base = []
    for b in map(json.loads, open(a.base, encoding="utf-8")):
        s, t = b["pair"].split("-")
        if "<SEG>" in b["text"] or (t in ("ko", "en") and s != t):
            continue
        base.append({"audio": b["audio"], "text": b["text"], "pair": b["pair"]})
    allr = base + rows
    random.Random(0).shuffle(allr)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in allr:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prefix rows {len(rows)} {kinds}; base rows kept {len(base)}; total {len(allr)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sents", "phrases", "hyprefix", "leak", "align", "build"])
    ap.add_argument("--en_manifest", default="models/zprobe/data/connective/manifest.jsonl")
    ap.add_argument("--ko_ref", default="models/zprobe/data/connective/hyref.jsonl")
    ap.add_argument("--x_manifest", default="models/zprobe/data/connective_en/manifest.jsonl")
    ap.add_argument("--sents"); ap.add_argument("--phrases"); ap.add_argument("--hy"); ap.add_argument("--aligned")
    ap.add_argument("--base"); ap.add_argument("--wav_dir"); ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_en", type=int, default=8000, help="English recordings kept for align/build (all FLEURS + CoVoST2 up to this)")
    ap.add_argument("--per_rec", type=int, default=3); ap.add_argument("--pad", type=float, default=0.2)
    a = ap.parse_args()
    if a.mode == "leak":
        a.hy, a.mode = a.phrases, "phrases"
    {"sents": sents, "phrases": phrases, "hyprefix": hyprefix, "align": align, "build": build}[a.mode](a)
