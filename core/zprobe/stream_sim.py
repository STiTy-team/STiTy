"""Streaming simulation for the translation adapter: re-translate growing audio prefixes and measure how the output
settles. Tests whether word-order differences make translation output unstable compared with transcription.

decode : for each utterance, decode on audio[0:t] for t = step, 2*step, ..., end (full re-decode each time, like a
         re-translation system). Writes {id, dur, steps: [[t, text], ...]} per line.
score  : per pair, from the decode files
   erasure     : characters deleted between consecutive outputs (len(prev) − common prefix), summed, / len(final)
   stable@x    : share of the final output that is already a stable prefix at x of the audio duration, where a
                 prefix is stable at time t when every later output (final included) starts with it

usage: PROBE_ADAPTER=... PYTHONPATH=Qwen3-ASR python core/zprobe/stream_sim.py decode --pair en-ko --ids X --limit 100 --out Y
       python core/zprobe/stream_sim.py score --files en-en=A.jsonl en-ko=B.jsonl ..."""
import argparse, json, os, sys, time
from pathlib import Path


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def decode(a):
    import librosa, torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import probe_z as pz
    src, tgt = a.pair.split("-")
    m = pz.load_model()
    tok = m.processor.tokenizer
    dev = next(m.model.thinker.parameters()).device
    rows = pz.read_rows(src)
    ids = [json.loads(l)["id"] for l in open(a.ids, encoding="utf-8")][: a.limit]
    prompt = m._build_text_prompt(context="", force_language=pz.LANGS[tgt][1])
    out = open(a.out, "w", encoding="utf-8"); t0 = time.time()
    for k, uid in enumerate(ids):
        wav, _ = librosa.load(str(pz.FLEURS / pz.LANGS[src][0] / "audio" / "test" / rows[uid][1]), sr=16000)
        dur = len(wav) / 16000
        cuts = [min(dur, a.step * (j + 1)) for j in range(int(-(-dur // a.step)))]
        steps = []
        for t in cuts:
            inputs = m.processor(text=[prompt], audio=[wav[: int(t * 16000)]], return_tensors="pt", padding=True)
            inputs = inputs.to(dev).to(torch.bfloat16)
            with torch.no_grad():
                o = m.model.generate(**inputs, max_new_tokens=160, no_repeat_ngram_size=a.no_repeat or None)
            seq = o.sequences if hasattr(o, "sequences") else o
            steps.append([round(t, 2), tok.decode(seq[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()])
        out.write(json.dumps({"id": uid, "dur": round(dur, 2), "steps": steps}, ensure_ascii=False) + "\n"); out.flush()
        if (k + 1) % 20 == 0:
            print(f"[{a.pair}] {k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)


def score(a):
    print("| pair | N | erasure / final length | stable@25% | stable@50% | stable@75% | stable before end |")
    print("|---|---|---|---|---|---|---|")
    for item in a.files:
        label, path = item.split("=", 1)
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        er, st = [], {0.25: [], 0.5: [], 0.75: [], "pre": []}
        for r in recs:
            texts = [s[1] for s in r["steps"]]; times = [s[0] for s in r["steps"]]
            final = texts[-1]
            if not final:
                continue
            er.append(sum(len(p) - lcp(p, c) for p, c in zip(texts, texts[1:])) / len(final))
            stable = []
            for j in range(len(texts)):
                s = len(texts[j])
                for later in texts[j + 1:]:
                    s = min(s, lcp(texts[j], later))
                stable.append(s / len(final))
            for x in (0.25, 0.5, 0.75):
                got = [stable[j] for j in range(len(texts)) if times[j] <= x * r["dur"] + 1e-6]
                st[x].append(max(got) if got else 0.0)
            st["pre"].append(max(stable[:-1]) if len(stable) > 1 else 0.0)
        mean = lambda v: sum(v) / len(v)
        print(f"| {label} | {len(er)} | {mean(er):.2f} | {mean(st[0.25]):.2f} | {mean(st[0.5]):.2f} | "
              f"{mean(st[0.75]):.2f} | {mean(st['pre']):.2f} |")


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["decode", "score"])
ap.add_argument("--pair"); ap.add_argument("--ids"); ap.add_argument("--out")
ap.add_argument("--limit", type=int, default=100); ap.add_argument("--step", type=float, default=1.0)
ap.add_argument("--no_repeat", type=int, default=0)
ap.add_argument("--files", nargs="*", default=[])
a = ap.parse_args()
decode(a) if a.mode == "decode" else score(a)
