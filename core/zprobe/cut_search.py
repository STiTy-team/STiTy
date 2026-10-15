"""Why does commit-and-continue lose quality on the translation adapter? Split "where the cut is" from "how the model
continues after a cut", with one cut per sentence on gold-timed English audio.

decode : for each sentence and each candidate cut k (after word k, at least --min_side words on both sides), translate
         audio[0:end_k] (unit 1), then continue on the full audio with unit 1 forced as the target prefix (unit 2).
         --strip removes a trailing sentence-final mark from unit 1 before it is forced (the adapter closes every
         partial translation with a period, then often stops). The SEG model's own cut (from seg_units.py label) is
         tagged so it can be read out.  -> {key "id:k", id, k, is_seg, hyp, units_tgt, ref_src, ref_tgt}
summary: reads COMET per key (comet_eval.py --per_sent) and the whole-sentence score, and prints
         seg cut / mean over all cuts (random cut) / best cut (oracle, chosen on the reference: an upper bound) / worst cut.

usage: PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/cut_search.py decode --ids ev_big2_en-ko.jsonl --units units_en.jsonl
         --tgt ko --limit 100 [--strip] --out X.jsonl
       python core/zprobe/cut_search.py summary --rows X.jsonl --scores S/label.jsonl --whole S/big2_en-ko.jsonl"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
NOSPACE = {"ja", "zh"}
FINAL = ".。!?！？"


def continuation(tok, input_ids, seq):
    """Text generated after a forced target prefix. Decoding only the new tokens breaks a Hangul syllable whose bytes
    straddle the prefix/new boundary ("���한"), so decode prefix+new together and cut the prefix text off."""
    start = int((input_ids == tok.convert_tokens_to_ids("<asr_text>")).nonzero()[-1]) + 1
    pre = tok.decode(input_ids[start:], skip_special_tokens=False)
    full = tok.decode(seq[start:], skip_special_tokens=False)
    full = full.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return full[len(pre):] if full.startswith(pre) else full[len(pre):]


def decode(a):
    import librosa, torch
    import probe_z as pz
    from qwen_asr import Qwen3ForcedAligner
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
    import csv
    rows = {r[0]: r for r in csv.reader(open(pz.FLEURS / "en_us" / f"{a.split}.tsv", encoding="utf-8"), delimiter="\t",
                                        quoting=csv.QUOTE_NONE) if len(r) >= 3}
    U = {r["id"]: r for r in map(json.loads, open(a.units, encoding="utf-8"))} if a.units else {}
    base = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])

    def gen(wav, prefix):
        inputs = m.processor(text=[base + prefix], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=120, no_repeat_ngram_size=4)
        seq = o.sequences if hasattr(o, "sequences") else o
        return re.sub(r"\s+", " ", continuation(tok, inputs["input_ids"][0], seq[0]).replace("<SEG>", " ")).strip()

    refs = [json.loads(l) for l in open(a.ids, encoding="utf-8")]
    U_ok = [r for r in refs if r["id"] in rows and U.get(r["id"], {}).get("aligned", True)][a.skip: a.skip + a.limit]
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time()
    for n, r in enumerate(U_ok):
        uid = r["id"]; gold = rows[uid][2].strip(); gw = gold.split()
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / a.split / rows[uid][1]), sr=16000)
        items = aligner.align((wav, 16000), gold, "English")[0]
        ends = [float(it.end_time) for it in items]
        scale = 1000.0 if ends and ends[-1] > 60 else 1.0
        ends = [e / scale for e in ends]
        if len(ends) != len(gw):
            continue
        u = U.get(uid, {"units": [gold]}); seg_k = len(u["units"][0].split()) if len(u["units"]) > 1 else None
        whole = gen(wav, "")
        out.write(json.dumps({"key": f"{uid}:0", "id": uid, "k": 0, "is_seg": False, "hyp": whole, "units_tgt": [whole, ""],
                              "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"]}, ensure_ascii=False) + "\n")
        for k in range(a.min_side, len(gw) - a.min_side + 1):
            u1 = gen(wav[: int(min(len(wav) / 16000, ends[k - 1] + a.pad) * 16000)], "")
            forced = u1.rstrip(FINAL).rstrip() if a.strip else u1
            sep = " <SEG> " if a.seg_prefix and forced else ("" if a.tgt in NOSPACE or not forced else " ")
            u2 = gen(wav, forced + sep)
            hyp = (forced + (" " if a.seg_prefix else sep) + u2) if a.strip or a.seg_prefix else ((u1 + u2) if a.tgt in NOSPACE else f"{u1} {u2}".strip())
            out.write(json.dumps({"key": f"{uid}:{k}", "id": uid, "k": k, "is_seg": k == seg_k, "hyp": hyp,
                                  "units_tgt": [u1, u2], "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"]},
                                 ensure_ascii=False) + "\n")
        out.flush()
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(U_ok)} {time.time() - t0:.0f}s", flush=True)


def summary(a):
    import collections
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
    sc = [json.loads(l) for l in open(a.scores, encoding="utf-8")]
    for r, s in zip(rows, sc):
        r["comet"] = s["comet"]
    whole = {r["id"]: r["comet"] for r in rows if r["k"] == 0}
    if a.whole and not whole:
        whole = {r["id"]: r["comet"] for r in map(json.loads, open(a.whole, encoding="utf-8"))}
    by = collections.defaultdict(list)
    for r in rows:
        if r["k"]:
            by[r["id"]].append(r)
    seg, mean, best, worst, w, empty2 = [], [], [], [], [], 0
    for uid, rs in by.items():
        cs = [r["comet"] for r in rs]
        mean.append(sum(cs) / len(cs)); best.append(max(cs)); worst.append(min(cs)); w.append(whole[uid])
        s = [r["comet"] for r in rs if r["is_seg"]]
        if s:
            seg.append((s[0], whole[uid]))
        empty2 += sum(not r["units_tgt"][1] for r in rs)
    f = lambda v: sum(v) / len(v)
    print(f"sentences {len(by)}, cut candidates {len(rows)}, empty unit 2: {empty2 / len(rows):.2f}")
    print(f"| cut | COMET | − whole sentence |\n|---|---|---|")
    print(f"| whole sentence (no cut) | {f(w):.3f} | |")
    if seg:
        print(f"| SEG model cut (n={len(seg)}) | {f([x for x, _ in seg]):.3f} | {f([x - y for x, y in seg]):+.3f} |")
    print(f"| random cut (mean over cuts) | {f(mean):.3f} | {f(mean) - f(w):+.3f} |")
    print(f"| best cut (oracle) | {f(best):.3f} | {f(best) - f(w):+.3f} |")
    print(f"| worst cut | {f(worst):.3f} | {f(worst) - f(w):+.3f} |")


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["decode", "summary"])
ap.add_argument("--ids"); ap.add_argument("--units"); ap.add_argument("--tgt"); ap.add_argument("--out")
ap.add_argument("--limit", type=int, default=100); ap.add_argument("--min_side", type=int, default=3)
ap.add_argument("--strip", action="store_true")
ap.add_argument("--seg_prefix", action="store_true", help="force the committed piece as 'piece <SEG> ' (models trained with --seg_full)")
ap.add_argument("--split", default="test"); ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--pad", type=float, default=0.0, help="audio after the cut word's end (s)")
ap.add_argument("--rows"); ap.add_argument("--scores"); ap.add_argument("--whole")
a = ap.parse_args()
decode(a) if a.mode == "decode" else summary(a)
