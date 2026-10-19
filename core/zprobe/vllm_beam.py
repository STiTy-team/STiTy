"""Beam search on the vLLM backend, the way vLLM 0.14 itself does it for the async engine
(vllm/entrypoints/openai/serving_engine.py beam_search): every output token is one round of `beam_width`
requests with max_tokens=1 and logprobs=2*beam_width, the audio re-sent each time, beams ranked by
cum_logprob / len(prompt+output)**length_penalty. The production server's SamplingParams(use_beam_search=True)
no longer exists in vLLM, so it silently falls back to greedy; this measures whether the replacement is usable.

The model must be a merged checkpoint (LoRA folded in), e.g. models/zprobe/merged/segconn2.
usage: PYTHONPATH=Qwen3-ASR python core/zprobe/vllm_beam.py --model M --ids ev_big2_en-ko.jsonl --out X.jsonl --beams 4"""
import argparse, asyncio, json, os, re, sys, time, uuid

sys.path.insert(0, os.path.dirname(__file__))


async def run(a):
    import librosa
    import probe_z as pz
    from qwen_asr import Qwen3ASRModel
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    m = Qwen3ASRModel.LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, max_new_tokens=a.max_tokens)
    eng, tok = m.model, m.processor.tokenizer
    eos = tok.eos_token_id
    stop_ids = {eos} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()}
    prompt = m._build_text_prompt(context="", force_language=pz.LANGS[a.tgt][1])
    ids0 = tok(prompt, add_special_tokens=False)["input_ids"]
    rows = pz.read_rows("en")

    def clean(ids):
        return re.sub(r"\s+", " ", tok.decode(ids, skip_special_tokens=True).replace("<SEG>", " ")).strip()

    async def one(p, sp):
        last = None
        async for o in eng.generate(p, sp, request_id=str(uuid.uuid4())):
            last = o
        return last

    async def greedy(wav):
        o = await one({"prompt": prompt, "multi_modal_data": {"audio": [wav]}},
                      SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
        return clean(o.outputs[0].token_ids)

    async def beam(wav, width):
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2 * width)
        live, done, steps = [([], 0.0)], [], 0
        key = lambda b: b[1] / ((len(ids0) + len(b[0])) ** a.length_penalty)
        for _ in range(a.max_tokens):
            outs = await asyncio.gather(*[one(TokensPrompt(prompt_token_ids=ids0 + g, multi_modal_data={"audio": [wav]}), sp)
                                          for g, _ in live])
            steps += 1
            cand = []
            for (g, lp), o in zip(live, outs):
                for t, obj in o.outputs[0].logprobs[0].items():
                    cand.append((g + [t], lp + obj.logprob))
            cand.sort(key=lambda b: b[1], reverse=True)
            live = []
            for g, lp in cand:
                if g[-1] in stop_ids:
                    done.append((g[:-1], lp))
                else:
                    live.append((g, lp))
                if len(live) == width:
                    break
            if len(done) >= width or not live:
                break
        done.extend(live)
        best = max(done, key=key)
        return clean(best[0]), steps

    refs = [json.loads(l) for l in open(a.ids, encoding="utf-8")][: a.limit or None]
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time()
    for k, r in enumerate(refs):
        wav, _ = librosa.load(str(pz.FLEURS / "en_us" / "audio" / "test" / rows[r["id"]][1]), sr=16000)
        s = time.perf_counter(); g = await greedy(wav); ms_g = (time.perf_counter() - s) * 1000
        s = time.perf_counter(); b, steps = await beam(wav, a.beams); ms_b = (time.perf_counter() - s) * 1000
        out.write(json.dumps({"id": r["id"], "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"], "greedy": g, "beam": b,
                              "ms_greedy": round(ms_g, 1), "ms_beam": round(ms_b, 1), "beam_steps": steps},
                             ensure_ascii=False) + "\n"); out.flush()
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(refs)} {time.time() - t0:.0f}s greedy {ms_g:.0f}ms beam {ms_b:.0f}ms", flush=True)
    print("VLLM_BEAM_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tgt", default="ko"); ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=128); ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--gpu_mem", type=float, default=0.35); ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))
