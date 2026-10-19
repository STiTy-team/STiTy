"""n-best upper bound: how good is the best translation the adapter can already produce?
For each FLEURS sentence in --ids the adapter (PROBE_ADAPTER) decodes the full audio once greedily, once with beam search
(--beams candidates) and --samples times with sampling. nbest_score.py then scores every candidate with COMET and reports
the oracle (best candidate per sentence) and rerankers that need no reference (CometKiwi, MBR).

usage: PROBE_MODEL=<koen-seg-mix> PROBE_ADAPTER=models/zprobe/adapters/segconn2 PYTHONPATH=Qwen3-ASR \
         python core/zprobe/nbest.py --ids models/zprobe/data/ev_big2_en-ko.jsonl --tgt ko --out N.jsonl"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(__file__))
import probe_z as pz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True); ap.add_argument("--tgt", default="ko"); ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=8); ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import librosa, torch
    torch.manual_seed(a.seed)
    m = pz.load_model(); tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    rows = pz.read_rows("en")
    refs = [json.loads(l) for l in open(a.ids, encoding="utf-8")][: a.limit or None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    out = open(a.out, "a", encoding="utf-8"); t0 = time.time()
    prompt = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])

    def dec(o, n_in):
        seq = o.sequences if hasattr(o, "sequences") else o
        return [re.sub(r"\s+", " ", tok.decode(s[n_in:], skip_special_tokens=True).replace("<SEG>", " ")).strip() for s in seq]

    for k, r in enumerate(refs):
        if r["id"] in done: continue
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / "test" / rows[r["id"]][1]), sr=16000)
        inputs = m.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            greedy = dec(m.model.generate(**inputs, max_new_tokens=128), n_in)[0]
            beam = dec(m.model.generate(**inputs, max_new_tokens=128, num_beams=a.beams, num_return_sequences=a.beams,
                                        do_sample=False), n_in)
            sample = dec(m.model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=a.top_p, num_return_sequences=a.samples), n_in)
        rec = {"id": r["id"], "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"], "greedy": greedy, "beam": beam, "sample": sample}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(refs)} {time.time() - t0:.0f}s", flush=True)
    print("NBEST_DONE", flush=True)


if __name__ == "__main__":
    main()
