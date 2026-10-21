"""Real-time streaming run of the translation adapter on bench's exported audio, so it can sit in the bench FLEURS table.

--mode prefix (adapters trained on build_prefix.py rows, no <SEG>): every decode continues the text written so far and
whatever it adds is shown at once and never revised, so the translation streams token by token. --mode seg is the
segconn2 behaviour described below. --mode alignatt needs no retraining (AlignAtt, Papi et al. 2023): the decode
continues the raw text written so far token by token and stops before the first token whose attention over the audio
peaks in the last --att_sec seconds of what has arrived (the model is reaching for speech it has not fully heard);
the tokens before it are shown and never revised. Run with PROBE_ATTN=eager so attention weights are returned.

The audio of each item arrives at wall-clock speed. Whenever at least --step seconds of new audio have arrived, the adapter
(trained with --seg_full, e.g. segconn2) translates audio[0:now] with the committed pieces forced as "p1 <SEG> p2 <SEG> ";
a piece before a new <SEG> is committed when the same piece came out at --agree consecutive decodes. When the audio ends
the rest is decoded on the full audio and committed. A slow decode (beam search) therefore means fewer decodes and later
commits, as it would on a live server.

Each commit records the audio fed so far (fed_sec, computation-unaware) and the wall time it was ready (wall_sec,
computation-aware), the same two clocks bench/seamless/seamless_score.py uses. Output rows follow that script's s2s.jsonl:
{utt_id, ok, text, segments: [[text, fed_sec, wall_sec], ...]}. Score with core/zprobe/stream_bench_score.py.

usage: PROBE_MODEL=<koen-seg-mix> PROBE_ADAPTER=models/zprobe/adapters/segconn2 PYTHONPATH=Qwen3-ASR \
         python core/zprobe/stream_bench.py --manifest <export>/manifest.jsonl --tgt ko --agree 2 [--beams 4] --out X.jsonl"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_units import continuation, join  # noqa: E402


def main(a):
    import soundfile as sf, torch
    import probe_z as pz
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    base = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])

    def gen(wav, pieces):
        prefix = "".join(p + " <SEG> " for p in pieces) if a.mode == "seg" else ("".join(p + " " for p in pieces) if a.tgt not in ("ja", "zh") else "".join(pieces))
        inputs = m.processor(text=[base + prefix], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=a.max_new_tokens, no_repeat_ngram_size=a.no_repeat or None,
                                 num_beams=a.beams, do_sample=False)
        seq = o.sequences if hasattr(o, "sequences") else o
        out = continuation(tok, inputs["input_ids"][0], seq[0])
        torch.cuda.synchronize()
        return out

    layers = [int(x) for x in a.att_layers.split(",")]
    audio_id = getattr(m.model.config, "thinker_config", m.model.config)
    audio_id = getattr(audio_id, "audio_token_id", 151646)

    def gen_att(wav, raw, final):
        """New raw text after `raw` (may contain <SEG>), cut before the first token that looks at the newest audio."""
        inputs = m.processor(text=[base + raw], audio=[wav], return_tensors="pt", padding=True).to(dev).to(torch.bfloat16)
        ids = inputs["input_ids"][0]; aud = (ids == audio_id).nonzero().squeeze(-1)
        with torch.no_grad():
            o = m.model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
                                 output_attentions=not final, return_dict_in_generate=True)
        new = o.sequences[0, ids.shape[0]:]
        keep = len(new)
        if not final and len(aud):
            last = max(1, int(round(a.att_sec * len(aud) / (len(wav) / 16000))))
            for j in range(len(new)):
                att = o.attentions[j]
                w = torch.stack([att[l][0, :, -1, :] for l in layers]).float().mean((0, 1))
                if int(w[aud].argmax()) >= len(aud) - last:
                    keep = j; break
        new = new[:keep]
        if len(new) and int(new[-1]) in (tok.eos_token_id, tok.pad_token_id):
            new = new[:-1]
        txt = continuation(tok, ids, torch.cat([ids, new]))
        while "\ufffd" in txt and len(new):  # never show half of a multi-byte character
            new = new[:-1]; txt = continuation(tok, ids, torch.cat([ids, new]))
        torch.cuda.synchronize()
        return txt

    items = [json.loads(l) for l in open(a.manifest, encoding="utf-8")][: a.limit or None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["utt_id"] for l in open(a.out, encoding="utf-8")}
    if a.mode == "alignatt":
        gen_att(sf.read(items[0]["wav"], dtype="float32")[0][:16000], "", False)  # warm-up, not timed
    else:
        gen(sf.read(items[0]["wav"], dtype="float32")[0][:16000], [])  # warm-up, not timed
    out = open(a.out, "a", encoding="utf-8"); t0 = time.time(); ndec = 0
    for n, it in enumerate(items):
        if it["utt_id"] in done:
            continue
        wav, sr = sf.read(it["wav"], dtype="float32")
        assert sr == 16000
        dur = len(wav) / sr
        pieces, segs, last, decodes, raw = [], [], None, 0, ""
        start = time.perf_counter(); fed = 0.0
        while True:
            now = time.perf_counter() - start
            if now < dur and (now < a.first or now - fed < a.step):
                time.sleep(max(0.0, min(dur, max(a.first, fed + a.step)) - now))
                continue
            fed = min(dur, now)
            final = fed >= dur
            if a.mode == "alignatt":
                add = gen_att(wav[: int(fed * sr)], raw, final); decodes += 1
                wall = time.perf_counter() - start
                raw += add
                shown = re.sub(r"\s+", " ", add.replace("<SEG>", " ")).strip()
                if shown:
                    pieces.append(shown); segs.append([shown, round(fed, 3), round(wall, 3)])
                if final:
                    break
                continue
            text = gen(wav[: int(fed * sr)], pieces); decodes += 1
            wall = time.perf_counter() - start
            if a.mode == "prefix":
                add = re.sub(r"\s+", " ", text.replace("<SEG>", " ")).strip()
                if add:
                    pieces.append(add); segs.append([add, round(fed, 3), round(wall, 3)])
                if final:
                    break
                continue
            if final:
                rest = re.sub(r"\s+", " ", text.replace("<SEG>", " ")).strip()
                if rest or not pieces:
                    pieces.append(rest); segs.append([rest, round(fed, 3), round(wall, 3)])
                break
            if "<SEG>" in text:
                piece = re.sub(r"\s+", " ", text.split("<SEG>")[0]).strip()
                if piece and (a.agree < 2 or piece == last):
                    pieces.append(piece); segs.append([piece, round(fed, 3), round(wall, 3)]); last = None
                else:
                    last = piece or None
            else:
                last = None
        ndec += decodes
        hyp = ""
        for p in pieces:
            hyp = join(hyp, p, a.tgt)
        out.write(json.dumps({"utt_id": it["utt_id"], "ok": True, "text": hyp, "segments": segs, "decodes": decodes,
                              "duration_sec": round(dur, 3)}, ensure_ascii=False) + "\n")
        out.flush()
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(items)} {time.time() - t0:.0f}s, {ndec} decodes", flush=True)
    print("STREAM_BENCH_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--tgt", default="ko"); ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.5, help="decode again once this much new audio has arrived (s)")
    ap.add_argument("--first", type=float, default=1.0, help="first decode once this much audio has arrived (s)")
    ap.add_argument("--agree", type=int, default=2); ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--no_repeat", type=int, default=4); ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=["seg", "prefix", "alignatt"], default="seg")
    ap.add_argument("--att_sec", type=float, default=0.5, help="alignatt: stop when attention peaks in this much newest audio")
    ap.add_argument("--att_layers", default="16,20,24,27", help="alignatt: layers whose attention is averaged")
    main(ap.parse_args())
