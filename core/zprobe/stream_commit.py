"""True streaming with the translation adapter deciding its own commits: the audio grows by --step seconds, the adapter
(trained with --seg_full, e.g. segconn2) translates audio[0:t] with the committed pieces forced as "p1 <SEG> p2 <SEG> ",
and whatever it writes before a <SEG> is committed and never revised. At the end of the audio the rest is decoded on the
full audio and committed. No SEG model, no gold timing: the adapter alone decides where to cut.

--agree 2 waits until the same piece comes out at two consecutive steps before committing it (local agreement), which
trades latency for fewer premature pieces.

Output rows match seg_units.py adapter ({id, hyp, units_tgt, commit, dur, ref_src, ref_tgt}), so comet_eval.py
--strip_seg and seg_units.py score work on them unchanged.

usage: PROBE_MODEL=<koen-seg-mix> PROBE_ADAPTER=models/zprobe/adapters/segconn2 PYTHONPATH=Qwen3-ASR \
         python core/zprobe/stream_commit.py --ids ev_big2_en-ko.jsonl --tgt ko --out X.jsonl [--step 0.5] [--agree 2]"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_units import continuation, join  # noqa: E402


def main(a):
    import librosa, torch
    import probe_z as pz
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    rows = pz.read_rows("en")
    base = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])

    def gen(wav, pieces):
        prefix = "".join(p + " <SEG> " for p in pieces)
        inputs = m.processor(text=[base + prefix], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=120, no_repeat_ngram_size=a.no_repeat or None)
        seq = o.sequences if hasattr(o, "sequences") else o
        return continuation(tok, inputs["input_ids"][0], seq[0])

    refs = [json.loads(l) for l in open(a.ids, encoding="utf-8")][: a.limit or None]
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time(); ncalls = 0
    for n, r in enumerate(refs):
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / "test" / rows[r["id"]][1]), sr=16000)
        dur = len(wav) / 16000
        pieces, commit, last, trace = [], [], None, []
        t = a.first
        while t < dur - 1e-6:
            s = time.perf_counter()
            text = gen(wav[: int(t * 16000)], pieces); ncalls += 1
            torch.cuda.synchronize(); ms = time.perf_counter() - s
            trace.append([round(t, 2), text.strip()])
            if "<SEG>" in text:
                piece = re.sub(r"\s+", " ", text.split("<SEG>")[0]).strip()
                if piece and (a.agree < 2 or piece == last):
                    pieces.append(piece); commit.append(round(t + ms, 3)); last = None
                else:
                    last = piece or None
            else:
                last = None
            t += a.step
        s = time.perf_counter()
        rest = re.sub(r"\s+", " ", gen(wav, pieces).replace("<SEG>", " ")).strip(); ncalls += 1
        torch.cuda.synchronize(); ms = time.perf_counter() - s
        if rest or not pieces:
            pieces.append(rest); commit.append(round(dur + ms, 3))
        hyp = ""
        for p in pieces:
            hyp = join(hyp, p, a.tgt)
        out.write(json.dumps({"id": r["id"], "hyp": hyp, "units_tgt": pieces, "commit": commit, "dur": round(dur, 2),
                              "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"], "trace": trace}, ensure_ascii=False) + "\n")
        out.flush()
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(refs)} {time.time() - t0:.0f}s, {ncalls} decodes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids"); ap.add_argument("--tgt", default="ko"); ap.add_argument("--out")
    ap.add_argument("--step", type=float, default=0.5, help="audio added per decode (s)")
    ap.add_argument("--first", type=float, default=1.0, help="first decode at this much audio (s)")
    ap.add_argument("--agree", type=int, default=1, help="2: commit only when the same piece comes out at two consecutive steps")
    ap.add_argument("--no_repeat", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args())
